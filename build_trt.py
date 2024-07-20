#!/usr/bin/env python3

import tensorrt as trt

import onnx
import onnxscript
from onnxconverter_common import float16

from onnxscript.onnx_opset import opset17 as op

import onnx_tensorrt.backend as backend
import torch
import numpy as np

from pointgpt.utils.config import cfg_from_yaml_file
from pointgpt.tools import builder

EXPORT_ONNX: bool = True
OPSET: int = 17

custom_opset = onnxscript.values.Opset(domain="torch.onnx", version=OPSET)


if False:
    @onnxscript.script(custom_opset)
    def aten_unflatten(self, dim, sizes):
        """unflatten(Tensor(a) self, int dim, SymInt[] sizes) -> Tensor(a)"""

        self_size = op.Shape(self)

        # if dim < 0:
        # PyTorch accepts negative dim as reversed counting
        self_rank = op.Size(self_size)
        dim = self_rank + dim

        head_start_idx = op.Constant(value_ints=[0])
        head_end_idx = op.Reshape(dim, op.Constant(value_ints=[1]))
        head_part_rank = op.Slice(self_size, head_start_idx, head_end_idx)

        tail_start_idx = op.Reshape(dim + 1, op.Constant(value_ints=[1]))
        # = sys.maxint, exactly 2^63 - 1 -> 64 bit int
        tail_end_idx = op.Constant(value_ints=[9223372036854775807])
        tail_part_rank = op.Slice(self_size, tail_start_idx, tail_end_idx)

        final_shape = op.Concat(head_part_rank, sizes, tail_part_rank, axis=0)

        # return op.Reshape(self, final_shape)
        return op.Reshape(self, final_shape)

    def custom_unflatten(g, self, dim, shape):
        return g.onnxscript_op(
            aten_unflatten, self, dim, shape).setType(
            self.type().with_sizes([1, 2, 3, 4, 5]))

    torch.onnx.register_custom_op_symbolic(
        symbolic_name="aten::unflatten",
        symbolic_fn=custom_unflatten,
        opset_version=OPSET,
    )


if False:
    @onnxscript.script(custom_opset)
    def ScaledDotProductAttention(
        query,
        key,
        value,
        dropout_p,
    ):
        # Swap the last two axes of key
        key_shape = op.Shape(key)
        key_last_dim = key_shape[-1:]
        key_second_last_dim = key_shape[-2:-1]
        key_first_dims = key_shape[:-2]
        # Contract the dimensions that are not the last two so we can transpose
        # with a static permutation.
        key_squeezed_shape = op.Concat(
            op.Constant(value_ints=[-1]),
            key_second_last_dim, key_last_dim, axis=0)
        key_squeezed = op.Reshape(key, key_squeezed_shape)
        key_squeezed_transposed = op.Transpose(key_squeezed, perm=[0, 2, 1])
        key_transposed_shape = op.Concat(
            key_first_dims,
            key_last_dim,
            key_second_last_dim,
            axis=0)
        key_transposed = op.Reshape(
            key_squeezed_transposed,
            key_transposed_shape)

        embedding_size = op.CastLike(op.Shape(query)[-1], query)
        # embedding_size = op.Shape(query)[-1]
        scale = op.Div(1.0, op.Sqrt(embedding_size))

        # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
        # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96
        # for math
        query_scaled = op.Mul(query, op.Sqrt(scale))
        key_transposed_scaled = op.Mul(key_transposed, op.Sqrt(scale))
        attn_weight = op.Softmax(
            op.MatMul(query_scaled, key_transposed_scaled),
            axis=-1,
        )
        # attn_weight, _ = op.Dropout(attn_weight, dropout_p)
        return op.MatMul(attn_weight, value)

    def custom_scaled_dot_product_attention(
            g, query, key, value, attn_mask, dropout, is_causal, scale=None):
        return g.onnxscript_op(
            ScaledDotProductAttention, query, key, value, dropout).setType(
            value.type())

    torch.onnx.register_custom_op_symbolic(
        symbolic_name="aten::scaled_dot_product_attention",
        symbolic_fn=custom_scaled_dot_product_attention,
        opset_version=OPSET,
    )


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pointgpt_cfg = cfg_from_yaml_file(
            "/tmp/PointGPT/pointgpt/cfgs/PointGPT-S/no-decoder.yaml")
        self.model = builder.model_builder(pointgpt_cfg.model)

    def forward(self,
                nbr: torch.Tensor,
                ctr: torch.Tensor):
        with torch.cuda.amp.autocast():
            return self.model.extract_embed_inner(nbr, ctr)[..., 1:, :]


INPUT_SHAPE = ((1024, 32, 32, 3),
               (1024, 32, 3))
OUTPUT_SHAPE = (1024, 32, 384)
onnx_filename = 'pointgpt.onnx'

if EXPORT_ONNX:
    model = MyModel()
    dummy_input = tuple([torch.randn(*s) for s in INPUT_SHAPE])
    model(*dummy_input)
# output: B G M 3
# center : B G 3

# Create an instance of the PyTorch model
# INPUT_SHAPE = (1024, 512, 3)

if EXPORT_ONNX:
    # Export the PyTorch model to ONNX
    dummy_input = tuple([torch.randn(*s) for s in INPUT_SHAPE])
    torch.onnx.export(model,
                      dummy_input,
                      onnx_filename,
                      verbose=True,
                      custom_opsets={'torch.onnx': OPSET},
                      input_names=['nbr', 'ctr'],
                      # output_names=['output'],
                      opset_version=OPSET
                      )

# Load the ONNX model
print('LOAD', flush=True)
model_onnx = onnx.load(onnx_filename)
# model_onnx = float16.convert_float_to_float16(model_onnx)

print('node 20?', model_onnx.graph.node[20])
print('node 20?', model_onnx.graph.node[19])
print('node name', model_onnx.graph.node[19].name)
print('node name', model_onnx.graph.node[19].output)
# print(model_onnx)

# Create a TensorRT builder and network
print('Builder')
builder = trt.Builder(trt.Logger(trt.Logger.INFO))
# explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
print('Create_network')
explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
network = builder.create_network(explicit_batch)
# network = builder.create_network()

# Create an ONNX-TensorRT backend
print('Create_parser', flush=True)
parser = trt.OnnxParser(network, builder.logger)
print('parse', flush=True)
parse_output = parser.parse(model_onnx.SerializeToString())
print('parse_output', parse_output,
      flush=True)
print('network 20?',
      network.get_layer(19).get_output(0).shape,
      network.get_layer(20).get_output(0).shape,
      'type', network.get_layer(20).type,
      'name', network.get_layer(20).name
      )
for i in range(network.num_layers):
    print(network.get_layer(i).type,
          network.get_layer(i).name)
# print('GLOT',
#       parser.get_layer_output_tensor('n20', 0).shape
#       )
print('??error??', flush=True)
for idx in range(parser.num_errors):
    print(parser.get_error(idx))
print('??error??')
if True:
    print('num layer?', network.num_layers)
    last_layer = network.get_layer(network.num_layers - 1)
    print('last_layer', last_layer)
    # Check if last layer recognizes it's output
    if not last_layer.get_output(0):
        # If not, then mark the output using TensorRT API
        print(last_layer.get_output(0))
        network.mark_output(last_layer.get_output(0))
print(network.num_inputs)
print(network.num_outputs)

print(network)

# Set up optimization profile and builder parameters
profile = builder.create_optimization_profile()
for k, v in zip(['nbr', 'ctr'], INPUT_SHAPE):
    profile.set_shape(k, v, v, v)
# profile.set_shape('output', OUTPUT_SHAPE,
#                   OUTPUT_SHAPE,
#                   OUTPUT_SHAPE)
builder_config = builder.create_builder_config()
# hmm~
# builder_config.set_flag(trt.BuilderFlag.FP16)

print(dir(builder_config))
# ???
# builder_config.max_workspace_size = 1 << 30
builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
# builder_config.set_flag(trt.BuilderFlag.SAFETY_SCOPE
# builder_config.flags = 1 << int(trt.BuilderFlag.STRICT_TYPES)

# Build the TensorRT engine from the optimized network
# engine = builder.build_engine(network, builder_config)
# print('hmm~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~',
#       parser.get_layer_output_tensor('n20', 0).shape
#       )
print('support?',
      builder.is_network_supported(network,
                                   builder_config)
      )
engine = builder.build_serialized_network(network, builder_config)
if True:
    try:
        with open('pointgpt.trt', "wb") as fp:
            fp.write(engine)
    except BaseException:
        with open('pointgpt.trt', 'wb') as fp:
            fp.write(engine.serialize())
    # Allocate device memory for input and output buffers
    input_name = 'input'
    output_name = 'output'
    input_shape = INPUT_SHAPE
    output_shape = OUTPUT_SHAPE
    input_buf = trt.cuda.alloc_buffer(
        builder.max_batch_size * sum([trt.volume(s) for s in INPUT_SHAPE]) *
        trt.float32.itemsize)
    output_buf = trt.cuda.alloc_buffer(
        builder.max_batch_size * sum([trt.volume(s) for s in OUTPUT_SHAPE]) *
        trt.float32.itemsize)

    # Create a TensorRT execution context
    context = engine.create_execution_context()

    # Run inference on the TensorRT engine
    input_data = tuple([torch.randn(*s).numpy()
                        for s in INPUT_SHAPE])
    output_data = np.empty(output_shape, dtype=np.float32)
    input_buf.host = input_data.ravel()
    trt_outputs = [output_buf.device]
    trt_inputs = [input_buf.device]
    context.execute_async_v2(
        bindings=trt_inputs + trt_outputs,
        stream_handle=trt.cuda.Stream())
    output_buf.device_to_host()
    output_data[:] = np.reshape(output_buf.host, output_shape)

    # Print the output
    print(output_data)
