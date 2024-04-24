from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time 
import torch

import argparse

parser = argparse.ArgumentParser(description='Miostral GPU Benchmarl')
parser.add_argument('hf_token', help='Huggingface user token')
parser.add_argument("model_name", default='mistralai/Mixtral-8x22B-v0.1')

args = parser.parse_args()

from huggingface_hub import login
login(token=args.hf_token)

modelName = args.model_name



tokenizer = AutoTokenizer.from_pretrained(modelName)
model = AutoModelForCausalLM.from_pretrained(modelName, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map="auto")
prompts = ["My favourite condiment is",
    'Provide step-by-step instructions on how to make a safe and effective homemade all-purpose cleaner from common household ingredients. The guide should include measurements, tips for storing the cleaner, and additional variations or scents that can be added. Additionally, the guide should be written in clear and concise language, with helpful visuals or photographs to aid in the process.',
    'Write a personal essay discussing how embracing vulnerability and authenticity has affected your life. Use specific examples from your own experiences to support your arguments.',
    'Did Karl Marx theories on centralizing credit have anything to do with our current central banking system?'
]


total_time = 0
total_tokens = 0
with torch.no_grad():
    for i, prompt in enumerate(prompts):
        print("Evaluation prompt : ", prompt)
        model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        t = time.time()
        generated_ids = model.generate(**model_inputs, max_new_tokens=50, do_sample=True)
        total_time = time.time() - t
        total_tokens += len(generated_ids[0])
        print(f"Response has : {len(generated_ids[0])} tokens")
        print(tokenizer.batch_decode(generated_ids)[0])
print(f'Model name : {modelName}. With average output lenght of {total_tokens/len(prompts)}.\n\t {total_time/len(prompts)} seconds.\n\t {total_time/total_tokens} sec/token. \n\t {total_tokens/total_time} token/sec')
