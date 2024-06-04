import os
os.environ['HF_HOME'] = 'hf_cache'

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import time 

from huggingface_hub import login
login(token='')


def compute_kv_cache_size(config, batch_size, sequence_length):
    """ Compute kv cache size. Taken from : https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices """
    d_model = config.hidden_size
    n_heads = config.num_attention_heads
    n_layers = config.num_hidden_layers
    n_kv_heads = config.num_key_value_heads
    # KV cache size calculation in bytes
    kv_cache_size = batch_size * sequence_length * (d_model // n_heads) * n_layers * 2 * 2 * n_kv_heads * 2  # 2 bytes for fp16
    return kv_cache_size

def estimate_mbu(model, batch_size, sequence_length, dt, peak_bandwidth=600e9):
    config = model.config
    total_param_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 2  # Each parameter is 2 bytes for fp16
    kv_cache_size = compute_kv_cache_size(config, batch_size, sequence_length)
    achieved_bandwidth = (total_param_size + kv_cache_size) / dt
    mbu = achieved_bandwidth / peak_bandwidth
    return mbu

def estimate_mfu(model, batch_size, dt, sequence_length, peak_flops=125e12):
    config = model.config
    N = sum(p.numel() for p in model.parameters() if p.requires_grad)
    L = config.num_hidden_layers
    H = config.num_attention_heads
    Q = config.hidden_size // H
    flops_per_token = 2 * N + 12 * L * H * Q * sequence_length
    flops_per_iter = flops_per_token * batch_size * sequence_length
    flops_achieved = flops_per_iter / dt
    mfu = flops_achieved / peak_flops
    return mfu


# Sample prompts.
prompts = [
    'Provide step-by-step instructions on how to make a safe and effective homemade all-purpose cleaner from common household ingredients. The guide should include measurements, tips for storing the cleaner, and additional variations or scents that can be added. Additionally, the guide should be written in clear and concise language, with helpful visuals or photographs to aid in the process.',
    'Write a personal essay discussing how embracing vulnerability and authenticity has affected your life. Use specific examples from your own experiences to support your arguments.',
    'Did Karl Marx theories on centralizing credit have anything to do with our current central banking system?'
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200)

# Create an LLM.
#llm = LLM(model="mistralai/Mistral-7B-v0.1", max_model_len=8192)
llm = LLM(model="meta-llama/Meta-Llama-3-8B", max_model_len=8192)

model_name = "meta-llama/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
for prompt in prompts :
    input_tokens = tokenizer(prompt, return_tensors='pt').input_ids.shape[1]
    start = time.time()
    response = llm.generate([prompt], sampling_params)
    end = time.time()
    generated_text = response[0].outputs[0].text
    latency = end-start
    output_tokens = len(response[0].outputs[0].token_ids)
    through_put = output_tokens / latency
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}, in {latency} sec. Throughput: {through_put}tokens/second")
    tpot = latency / output_tokens

    total_sequence_length = input_tokens + output_tokens
    # Calculate MBU
    mbu = estimate_mbu(model, 1, 200, tpot)
    print(f"Memory Bandwidth Utilization (MBU): {mbu}")

    # Estimate MFU

    mfu = estimate_mfu(model, 1, through_put, 200)
    print(f"Model FLOPs Utilization (MFU): {mfu}")

