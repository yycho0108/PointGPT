#!/usr/bin/env python3

from typing import Optional, Union, Sequence, Dict
import torch
import time
import numpy as np
import tensorrt as trt
import torch as th
import torch.nn as nn


class TRTWrapper(nn.Module):
    def __init__(self, engine: Union[str, trt.ICudaEngine] = 'pointgpt.trt',
                 output_names: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.engine = engine
        if isinstance(self.engine, str):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                with open(self.engine, mode='rb') as f:
                    engine_bytes = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        names = [_ for _ in self.engine]
        # input_names = list(filter(self.engine.binding_is_input, names))
        input_names = ['ctr', 'nbr']
        self._input_names = input_names
        # self._output_names = output_names
        self._output_names = output_names

        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    def forward(self, inputs: Dict[str, torch.Tensor]):
        engine = self.engine
        context = self.context

        bindings = [None] * (len(self._input_names) + len(self._output_names))
        indices = {self.engine.get_tensor_name(
            i): i for i in range(len(bindings))}
        for input_name, input_tensor in inputs.items():
            # idx = self.engine.get_tensor_location_index(input_name)
            idx = indices[input_name]
            # self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

        outputs = {}
        for output_name in self._output_names:
            # idx = self.engine.get_binding_index(output_name)
            idx = indices[output_name]
            dtype = torch.float32
            shape = tuple(self.engine.get_tensor_shape(output_name))
            device = torch.device('cuda')
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()
        # t0 = time.time()
        # for i in range(20):
        num_io = engine.num_io_tensors
        for i in range(num_io):
            context.set_tensor_address(engine.get_tensor_name(i),
                                       bindings[i])
        self.context.execute_async_v3(
            # bindings,
            stream_handle=torch.cuda.current_stream().cuda_stream
        )
        # t1 = time.time()
        # print('trt model time: ', t1 - t0)
        return outputs


def main():
    net = TRTWrapper()
    inputs = {
        'nbr': th.zeros(size=(1024, 32, 32, 3),
                        dtype=th.float32,
                        device='cuda'),
        'ctr': th.zeros(size=(1024, 32, 3),
                        dtype=th.float32,
                        device='cuda')
    }
    dts = []
    s = th.cuda.Stream()
    with th.cuda.stream(s):
        for _ in range(128):
            t0 = time.time()
            output = net(inputs)
            t1 = time.time()
            dts.append(t1 - t0)
    print(dts[1:])
    print(np.mean(dts[1:]))


if __name__ == '__main__':
    main()
