#!/usr/bin/env python3
import ctypes
from typing import Optional, Union
import torch
import time
import numpy as np
import tensorrt as trt
# import pycuda.autoinit
# import pycuda.driver as cuda
from cuda import cuda, cudart


def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""

    def __init__(self, size: int, dtype: Optional[np.dtype] = None):
        dtype = dtype or np.dtype(np.uint8)
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(
            ctypes.cast(host_mem, pointer_type), (size,))
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, data: Union[np.ndarray, bytes]):
        if isinstance(data, np.ndarray):
            if data.size > self.host.size:
                raise ValueError(
                    f"Tried to fit an array of size {data.size} into host memory of size {self.host.size}"
                )
            np.copyto(self.host[:data.size], data.flat, casting='safe')
        else:
            assert self.host.dtype == np.uint8
            self.host[:self.nbytes] = np.frombuffer(data, dtype=np.uint8)

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))


def allocate_buffers(engine: trt.ICudaEngine,
                     profile_idx: Optional[int] = None):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda_call(cudart.cudaStreamCreate())
    tensor_names = [engine.get_tensor_name(i)
                    for i in range(engine.num_io_tensors)]
    for binding in tensor_names:
        # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
        # Pick out the max shape to allocate enough memory for the binding.
        shape = engine.get_tensor_shape(
            binding) if profile_idx is None else engine.get_tensor_profile_shape(binding, profile_idx)[-1]
        shape_valid = np.all([s >= 0 for s in shape])
        if not shape_valid and profile_idx is None:
            raise ValueError(f"Binding {binding} has dynamic shape, " +
                             "but no profile was specified.")
        size = trt.volume(shape)
        trt_type = engine.get_tensor_dtype(binding)

        # Allocate host and device buffers
        try:
            dtype = np.dtype(trt.nptype(trt_type))
            bindingMemory = HostDeviceMem(size, dtype)
        # no numpy support: create a byte array instead (BF16, FP8, INT4)
        except TypeError:
            size = int(size * trt_type.itemsize)
            bindingMemory = HostDeviceMem(size)

        # Append the device buffer to device bindings.
        # bindings.append(int(bindingMemory.device))
        bindings.append(bindingMemory.device)

        # Append to the appropriate list.
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(bindingMemory)
        else:
            outputs.append(bindingMemory)
    return inputs, outputs, bindings, stream


def _do_inference_base(inputs, outputs, stream, execute_async_func):
    # Transfer input data to the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    #[cuda_call(cudart.cudaMemcpyAsync(inp.device, inp.host,
    #           inp.nbytes, kind, stream)) for inp in inputs]
    # Run inference.
    execute_async_func()
    # Transfer predictions back from the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    #[cuda_call(cudart.cudaMemcpyAsync(out.host, out.device,
    #           out.nbytes, kind, stream)) for out in outputs]
    # Synchronize the stream
    # cuda_call(cudart.cudaStreamSynchronize(stream))
    # Return only the host outputs.
    return [out.host for out in outputs]


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, engine, bindings, inputs, outputs, stream):
    def execute_async_func():
        context.execute_async_v3(stream_handle=stream)
    # Setup context tensor address.
    num_io = engine.num_io_tensors
    for i in range(num_io):
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
    return _do_inference_base(inputs, outputs, stream, execute_async_func)


def main():
    with open("pointgpt.trt", "rb") as fp:
        engine_data = fp.read()
    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)

    INPUT_SHAPE = ((1024, 32, 32, 3),
                   (1024, 32, 3))

    OUTPUT_SHAPE = (1024, 32, 384)
    # Allocate device memory for input and output buffers
    output_name = 'output'
    input_shape = INPUT_SHAPE
    output_shape = OUTPUT_SHAPE

    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # input_buf = trt.cuda.alloc_buffer(
    #     builder.max_batch_size * sum([trt.volume(s) for s in INPUT_SHAPE]) *
    #     trt.float32.itemsize)
    # output_buf = trt.cuda.alloc_buffer(
    #     builder.max_batch_size * sum([trt.volume(s) for s in OUTPUT_SHAPE]) *
    #     trt.float32.itemsize)

    # Create a TensorRT execution context
    context = engine.create_execution_context()

    # Run inference on the TensorRT engine
    # input_data = tuple([torch.randn(*s).numpy()
    #                     for s in INPUT_SHAPE])
    # output_data = np.empty(output_shape, dtype=np.float32)
    # inputs.host = input_data.ravel()
    # trt_outputs = [output_buf.device]
    # trt_inputs = [inputs.device]

    dts = []
    for _ in range(128):
        for x in inputs:
            x.host = np.random.normal(size=x.host.shape).astype(
                dtype=x.host.dtype)
        # host = np.random.normal(size=inputs.host.shape)
        t0 = time.time()
        [output] = do_inference(
            context,
            engine,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream)
        t1 = time.time()
        dts.append(t1 - t0)
    cuda_call(cudart.cudaStreamSynchronize(stream))
    t2 = time.time()
    dts.append(t2 - t1)
    print('dts', dts[1:])
    print('avg. dt', np.mean(dts[1:]))
    print(output)

    # context.execute_async_v2(
    #     bindings=trt_inputs + trt_outputs,
    #     stream_handle=trt.cuda.Stream())
    # output_buf.device_to_host()
    # output_data[:] = np.reshape(output_buf.host, output_shape)

    # Print the output
    # print(output_data)


if __name__ == '__main__':
    main()
