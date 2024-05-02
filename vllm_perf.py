import os
os.environ['HF_HOME'] = 'hf_cache' #from s3 or attached volume to avoid download

from vllm import LLM, SamplingParams
import time 

from huggingface_hub import login
login(token='')

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


# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
for prompt in prompts :
    start = time.time()
    response = llm.generate([prompt], sampling_params)
    end = time.time()
    generated_text = response[0].outputs[0].text
    latency = end-start
    output_tokens = len(response[0].outputs[0].token_ids)
    through_put = output_tokens / latency
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}, in {latency} sec. Throughput: {through_put}tokens/second")
