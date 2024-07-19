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

from typing import Optional

import torch

from ..tensor import Optimizer, QBytesTensor, qtype, quantize_activation
from .qmodule import QModuleMixin, register_qmodule

# This is required to be able to access `torch.ops.quanto_ext.*` members defined in C++ through `TORCH_LIBRARY`. 
from optimum.quanto.library.ext.cuda import ext  # noqa: F401

__all__ = ["QLinear"]

def _forward_linear(input, other, bias):
    if False and other.__class__.__name__ == "AWQBitsTensor":
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
    elif False and other.__class__.__name__ == "TinyGemmQBitsTensor":
        if type(input) != torch.Tensor:
            input = input.dequantize()
        in_features = input.shape[-1]
        out_features = other.shape[0]
        output_shape = input.shape[:-1] + (out_features,)
        output = torch._weight_int4pack_mm(
            input.view(-1, in_features), other._data._data, other._group_size, other._scale_shift
        )
        output = output.view(output_shape)
    elif True or other.__class__.__name__ == "MarlinF8QBytesTensor":
        input_shape = input.shape

        if input.ndim > 2:
            input = input.view(-1, input_shape[-1])
        
        # print("other._data", other._data.shape, other._data.dtype)
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


@register_qmodule(torch.nn.Linear)
class QLinear(QModuleMixin, torch.nn.Linear):
    @classmethod
    def qcreate(
        cls, module, weights: qtype, activations: Optional[qtype] = None, optimizer: Optional[Optimizer] = None
    ):
        return cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            dtype=module.weight.dtype,
            device=module.weight.device,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
        )

    def qforward(self, input: torch.Tensor) -> torch.Tensor:
        if self.activation_qtype is not None and not isinstance(input, QBytesTensor):
            # Quantize activations to be able to take advantage of accelerated matmul
            input = quantize_activation(input, qtype=self.activation_qtype, scale=self.input_scale)
        # We always use quantized weights
        return _forward_linear(input, self.qweight, bias=self.bias)
