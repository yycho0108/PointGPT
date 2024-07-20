#!/usr/bin/env python3

from build_trt import MyModel
import time
import torch as th
#import torch_tensorrt
import numpy as np

INPUT_SHAPE = ((1024, 32, 32, 3),
               (1024, 32, 3))
model = MyModel().to(device='cuda:0')
model.model.encoder = th.compile(model.model.GPT_Transformer.encoder)
for k,v in model.named_parameters():
    print(k, v.shape)
#model = th.compile(model, backend='tensorrt')
dummy_input = tuple([th.randn(*s, device='cuda:0') for s in INPUT_SHAPE])
dts = []
with th.cuda.amp.autocast(enabled=True):
    for _ in range(128):
        t0 = time.time()
        model(*dummy_input)
        t1 = time.time()
        dts.append(t1 - t0)
print(dts[1:])
print(np.mean(dts[1:]))
