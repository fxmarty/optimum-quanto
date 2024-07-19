# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import torch

# from .marlin import MarlinF8QBytesTensor
# from .qbits import AWQBitsTensor, TinyGemmQBitsTensor
# from .qbytes import QBytesTensor
# from .fallback import qfallback

# This is required to be able to access `torch.ops.quanto_ext.*` members defined in C++ through `TORCH_LIBRARY`. 
from optimum.quanto.library.ext.cuda import ext  # noqa: F401


__all__ = ["get_qtensor_func", "register_qtensor_func"]


_QTENSOR_FUNC_TABLE = {}


def register_qtensor_func(funcs):
    """
    Used for registering a new __torch_dispatch__ function to QTensor.

    The code to register a new function looks like:

    @register_qtensor_func(list_of_funcs)
    def foo(func, *args, **kwargs):
        <implementation>
    """

    def wrapper(qfunc):
        for func in funcs:
            _QTENSOR_FUNC_TABLE[func] = partial(qfunc, func)

    return wrapper


def get_qtensor_func(func):
    return _QTENSOR_FUNC_TABLE.get(func, None)


@register_qtensor_func([torch._has_compatible_shallow_copy_type])
def has_compatible_shallow_copy_type(func, input: torch.Tensor, from_: torch.Tensor):
    # Prevent torch from trying to shallow copy one QTensor to another
    return False


class QTensorLinear(torch.autograd.Function):
    """Quantized linear function.

    This is a quantized implementation of torch.nn.functional.linear.

    It defines explicitly the backward pass instead of letting pytorch
    build it by combining the gradients of the underlying quantized operations.

    This has two main benefits:

    - this saves computations,
    - this allows to use operations that do not have a registered backward method,
    such as quanto custom operations.

    The drawback is that the extra tensors involved in the quantization graph, such as
    the scales and shift, cannot be trained.
    This is however consistent with the quanto quantizers backward pass, that returns
    a zero gradient for these tensors.
    """

    @staticmethod
    def forward(ctx, input, other, bias):
        ctx.save_for_backward(input, other)
        return QTensorLinear._forward(input, other, bias)
        

    @staticmethod
    def _forward(input, other, bias):
        if other.__class__.__name__ == "AWQBitsTensor":
            if type(input) != torch.Tensor:
                input = input.dequantize()
            out_features, in_features = other.shape
            rows = input.numel() // in_features
            output = torch.ops.quanto.gemm(
                input,
                other._data._data,
                other._scale,
                other._shift,
                rows=rows,
                out_cols=out_features,
                in_cols=in_features,
                bits=4,
                group_size=other._group_size,
            )
        elif other.__class__.__name__ == "TinyGemmQBitsTensor":
            if type(input) != torch.Tensor:
                input = input.dequantize()
            in_features = input.shape[-1]
            out_features = other.shape[0]
            output_shape = input.shape[:-1] + (out_features,)
            output = torch._weight_int4pack_mm(
                input.view(-1, in_features), other._data._data, other._group_size, other._scale_shift
            )
            output = output.view(output_shape)
        elif other.__class__.__name__ == "MarlinF8QBytesTensor":
            input_shape = input.shape

            if input.ndim > 2:
                input = input.view(-1, input_shape[-1])
            
            output = torch.ops.quanto_ext.fp8_marlin_gemm(
                input,
                b_q_weight=other._data,
                b_scales=other._scale,  # .to(input.dtype)
                workspace=other._workspace,
                num_bits=8,
                size_m=input.shape[0],
                size_n=other._scale.shape[1],
                size_k=input.shape[1],
            )

            if len(input_shape) > 2:
                output = output.reshape(input_shape[:-1] + (other._scale.shape[1],))
        elif other.__class__.__name__ == "QBytesTensor":
            if input.__class__.__name__ == "QBytesTensor":
                output = torch.ops.quanto.qbytes_mm(input._data, other._data, input._scale * other._scale)
            else:
                output = torch.ops.quanto.qbytes_mm(input, other._data, other._scale)
        else:
            output = torch.matmul(input, other.t())
        if bias is not None:
            output = output + bias
        return output

    def backward(ctx, gO):
        input_gO = other_gO = bias_gO = None
        input, other = ctx.saved_tensors
        out_features, in_features = other.shape
        if ctx.needs_input_grad[0]:
            # grad(A@(B.t()) = gO => grad(A) = gO@(B.t().t()) = gO@B
            input_gO = torch.matmul(gO, other)
        if ctx.needs_input_grad[1]:
            # grad(B@A.t()) = gO.t() => grad(B) = gO.t()@(A.t().t()) = gO.t()@A
            other_gO = torch.matmul(gO.view(-1, out_features).t(), input.view(-1, in_features))
        if ctx.needs_input_grad[2]:
            # Bias gradient is the sum on all dimensions but the last one
            dim = tuple(range(gO.ndim - 1))
            bias_gO = gO.sum(dim)
        return input_gO, other_gO, bias_gO


@register_qtensor_func([torch.nn.functional.linear])
def linear(func, input, other, bias=None):
    if input.requires_grad:
        return QTensorLinear.apply(input, other, bias)
    else:
        return QTensorLinear._forward(input, other, bias)
