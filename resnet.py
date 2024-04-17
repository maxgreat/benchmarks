import os
os.environ['LRU_CACHE_CAPACITY'] = '1024'
os.environ['DNNL_DEFAULT_FPMATH_MODE'] = 'BF16'


import torch
from torchvision import models
import time
import tqdm

import argparse

parser = argparse.ArgumentParser("Simple Benchmark")
parser.add_argument("-bs", help="Batch Size", nargs='+', type=int, default=[1, 2, 3, 4, 8 , 12, 16])
parser.add_argument("--it", help="Iterations", type=int, default=100)
args = parser.parse_args()


for batch_size in args.bs:
    sample_input = [torch.rand(batch_size, 3, 224, 224)]
    eager_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = torch.jit.script(eager_model, example_inputs=[sample_input, ])

    model = model.eval()
    model = torch.jit.optimize_for_inference(model)

    with torch.no_grad():
        # warm up
        for _ in range(10):
            model(*sample_input)
        
        s = 0
        for i in tqdm.tqdm(range(args.it)):
            t = time.time()
            model(*sample_input)
            s += time.time() - t
        print(f"Resnet latency for batch size {batch_size} :", s/args.it)
