#!/usr/bin/env python3

from build_trt import MyModel
import time
import torch as th
#import torch_tensorrt
import numpy as np
from pkm.util.torch_util import set_seed
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

INPUT_SHAPE = ((3*1024, 32, 32, 3),
               (3*1024, 32, 3))
model = MyModel().to(device='cuda')
model.eval()
#model.model.encoder = th.compile(model.model.GPT_Transformer.encoder)
for k,v in model.named_parameters():
    print(k, v.shape)
#model = th.compile(model, backend='tensorrt')
# dummy_input = tuple([th.randn(*s, device='cuda:0') for s in INPUT_SHAPE])
set_seed(0)
inputs = {
        'nbr': th.randn(size=(3*1024, 32, 32, 3),
                        dtype=th.float32,
                        device='cuda'),
        'ctr': th.randn(size=(3*1024, 32, 3),
                        dtype=th.float32,
                        device='cuda')
}
dts = []
#with th.cuda.amp.autocast(enabled=True):
for i in range(128):
    t0 = time.time()
    with th.no_grad():
        output = model(inputs['nbr'][:8], inputs['ctr'][:8])
    if i == 0:
        print(output)
    t1 = time.time()
    dts.append(t1 - t0)
print(dts[1:])
print(np.mean(dts[1:]))
