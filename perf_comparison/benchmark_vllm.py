import argparse
import json
import random
import time
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

def estimate_mfu(nb_parameters, nb_hidden, hidden_size, nb_heads, batch_size, tput, sequence_length, peak_flops=125e12):
    Q = hidden_size // nb_heads
    flops_per_token = 2 * nb_parameters + 4 * nb_hidden * nb_heads * Q * sequence_length
    return (tput * flops_per_token) / peak_flops

def sample_requests(
    tokenizer: PreTrainedTokenizerBase,
    num_requests: int,
    input_length :int,
    fixed_output_len: int,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    from datasets import load_dataset
    d = load_dataset("wikipedia", "20220301.en").shuffle()

    # Filter out the conversations with less than 2 turns.
    dataset = d['train'][:num_requests]['text']

    filtered_dataset = []
    for prompt in dataset:
        # Tokenize the prompts and completions.
        prompt_token_ids = tokenizer(prompt).input_ids
        if(len(prompt_token_ids) <= 100):
            continue
        prompt_token_ids = prompt_token_ids[:input_length]
        completion = tokenizer(prompt).input_ids[input_length:input_length+fixed_output_len]
        prompt_len = len(prompt_token_ids)
        output_len = len(completion)
        filtered_dataset.append((tokenizer.decode(prompt_token_ids), prompt_len, output_len))

    return filtered_dataset


def run_vllm(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    quantization_param_path: Optional[str],
    device: str,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float = 0.9
) -> float:
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        quantization_param_path=quantization_param_path,
        device=device,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    # Add the requests to the engine.
    prompts = []
    sampling_params = []
    for prompt, _, output_len in requests:
        prompts.append(prompt)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=0.0 if use_beam_search else 1.0,
                top_p=1.0,
                use_beam_search=use_beam_search,
                ignore_eos=True,
                max_tokens=output_len,
            ))

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    end = time.perf_counter()
    return end - start, outputs


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)

    requests = sample_requests(tokenizer,
                                args.num_prompts, 
                                args.input_len,
                                args.output_len)

    elapsed_time, ouputs = run_vllm(
        requests, args.model, args.tokenizer, args.quantization,
        args.tensor_parallel_size, args.seed, args.n, args.use_beam_search,
        args.trust_remote_code, args.dtype, args.max_model_len,
        args.enforce_eager, args.kv_cache_dtype,
        args.quantization_param_path, args.device,
        args.enable_prefix_caching, args.enable_chunked_prefill,
        args.max_num_batched_tokens, args.gpu_memory_utilization)

    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len in requests)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")

    print("Time per token :", elapsed_time / total_num_tokens)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)

    mfu = estimate_mfu(nb_parameters=model.num_parameters(), 
                            nb_hidden=model.config.num_hidden_layers, 
                            hidden_size=model.config.hidden_size, 
                            nb_heads=model.config.num_attention_heads, 
                            batch_size=1, 
                            sequence_length=total_num_tokens, 
                            tput = total_num_tokens / elapsed_time, 
                            peak_flops=125e12)
        
    print(f"mfu : {mfu*100} %")
        

        

    # Output JSON results if specified
    if args.output_json:
        results = {
            "elapsed_time": elapsed_time,
            "num_requests": len(requests),
            "total_num_tokens": total_num_tokens,
            "requests_per_second": len(requests) / elapsed_time,
            "tokens_per_second": total_num_tokens / elapsed_time,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--input-len",
                        type=int,
                        default=100,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=300,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=[*QUANTIZATION_METHODS, None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=100,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.9,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument(
        '--kv-cache-dtype',
        type=str,
        choices=['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'],
        default="auto",
        help='Data type for kv cache storage. If "auto", will use model '
        'data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. '
        'ROCm (AMD GPU) supports fp8 (=fp8_e4m3)')
    parser.add_argument(
        '--quantization-param-path',
        type=str,
        default=None,
        help='Path to the JSON file containing the KV cache scaling factors. '
        'This should generally be supplied, when KV cache dtype is FP8. '
        'Otherwise, KV cache scaling factors default to 1.0, which may cause '
        'accuracy issues. FP8_E5M2 (without scaling) is only supported on '
        'cuda version greater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is '
        'instead supported for common inference criteria.')
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help='device type for vLLM execution, supporting CUDA and CPU.')
    parser.add_argument(
        "--enable-prefix-caching",
        action='store_true',
        help="enable automatic prefix caching for vLLM backend.")
    parser.add_argument("--enable-chunked-prefill",
                        action='store_true',
                        help="enable chunked prefill for vLLM backend.")
    parser.add_argument('--max-num-batched-tokens',
                        type=int,
                        default=None,
                        help='maximum number of batched tokens per '
                        'iteration')
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the throughput results in JSON format.')
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    main(args)
