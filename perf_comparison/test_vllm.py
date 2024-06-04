import os
os.environ['HF_HOME'] = '/home/ubuntu/data/hf_cache'

from vllm import LLM, SamplingParams
import time 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from utilization import *

from huggingface_hub import login
login(token='')

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=500)

# Create an LLM.
#llm = LLM(model="mistralai/Mistral-7B-v0.1", max_model_len=8192)
llm = LLM(model="meta-llama/Meta-Llama-3-8B", max_model_len=8192, dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", torch_dtype=torch.float16)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
start = time.time()
response = llm.generate(prompts, sampling_params)
end = time.time()
generated_text = response[0].outputs[0].text
latency = end-start
"""
output_tokens = len(response[0].outputs[0].token_ids)
through_put = output_tokens / latency
print(f"Generated text: {generated_text!r}, in {latency} sec. Throughput: {through_put}tokens/second")
"""
for output in response:
    print(len(output.outputs))
    ftl = output.metrics.first_token_time - output.metrics.first_scheduled_time 
    TPOT = ( output.metrics.finished_time - output.metrics.first_token_time ) / len(output.outputs[0].token_ids)
    tput = len(output.outputs[0].token_ids) / ( output.metrics.finished_time - output.metrics.first_scheduled_time)
    print(f"ftl: {ftl!r} sec, TPOT: {TPOT!r} sec/token , tput: {tput}token/sec")
    mbu = estimate_mbu(total_param_size=model.num_parameters() * 2, 
                        nb_hidden=model.config.num_hidden_layers, 
                        hidden_size=model.config.hidden_size, 
                        nb_heads=model.config.num_attention_heads, 
                        nb_kv_heads=model.config.num_key_value_heads, 
                        batch_size=1, 
                        sequence_length=len(output.outputs[0].token_ids) + len(output.prompt_token_ids), 
                        dt = output.metrics.finished_time - output.metrics.first_token_time, 
                        peak_bandwidth=600e9)
    
    print(f"mbu : {mbu} %")

    mfu = estimate_mfu(nb_parameters=model.num_parameters(), 
                        nb_hidden=model.config.num_hidden_layers, 
                        hidden_size=model.config.hidden_size, 
                        nb_heads=model.config.num_attention_heads, 
                        batch_size=1, 
                        sequence_length=len(output.outputs[0].token_ids) + len(output.prompt_token_ids), 
                        dt = output.metrics.finished_time - output.metrics.first_token_time, 
                        peak_flops=125e12)
    
    print(f"mfu : {mfu} %")
