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

import ast
import torch

from ..qbytes import QBytesTensor
from ..qtype import qtypes

# This is required to be able to access `torch.ops.quanto_ext.*` members defined in C++ through `TORCH_LIBRARY`. 
from optimum.quanto.library.ext.cuda import ext  # noqa: F401


def pack_fp8_as_int32(fp8_tensor: torch.Tensor) -> torch.Tensor:
    """
    Repack FP8 weights to gptq format (packed int32 elements).
    """
    assert fp8_tensor.dtype == torch.float8_e4m3fn

    if fp8_tensor.shape[0] % 4 != 0:
        raise ValueError(f"Leading tensor dimension is not divisable by 4: {fp8_tensor.shape[0]}")

    # Reshape to prepare for packing
    reshaped = fp8_tensor.reshape(-1, 4, *fp8_tensor.shape[1:])

    # Convert fp8 to uint8 (byte) representation
    byte_tensor = reshaped.view(torch.uint8)

    # Pack 4 uint8 values into one int32
    packed = torch.zeros(
        fp8_tensor.shape[0] // 4,
        fp8_tensor.shape[1],
        dtype=torch.int32,
        device=fp8_tensor.device,
    )

    for i in range(4):
        packed.bitwise_or_(byte_tensor[:, i].to(torch.int32) << i * 8)

    return packed


class MarlinF8QBytesTensor(QBytesTensor):
    @staticmethod
    def __new__(cls, qtype, axis, size, stride, data, scale, zeropoint, requires_grad=False):
        assert data.device.type == "cuda"
        assert data.device == scale.device
        return torch.Tensor._make_wrapper_subclass(
            cls, size, strides=stride, dtype=scale.dtype, device=data.device, requires_grad=requires_grad
        )

    def __init__(self, qtype, axis, size, stride, data, scale, zeropoint, requires_grad=False):
        if requires_grad:
            raise NotImplementedError("Backward with Marlin FP8 is not implemented.")

        assert axis is None
        assert data.ndim == 2

        # When freezing (`model.freeze()`), the data is already packed on int32.
        if data.dtype != torch.int32:
            out_features, in_features = data.shape
            self._workspace = torch.zeros(out_features // 64 * 16, dtype=torch.int, device=data.device)

            scale = scale.repeat(1, out_features).to(data.device)
            data_int32 = pack_fp8_as_int32(data.T)
            perm = torch.empty(0, dtype=torch.int, device=data.device)

            data_repack = torch.ops.quanto_ext.gptq_marlin_repack(
                b_q_weight=data_int32, perm=perm, size_k=in_features, size_n=out_features, num_bits=8
            )
        elif data.dtype == torch.int32 and scale.ndim == 2:
            data_repack = data

            out_features = data_repack.shape[1] // 4
            self._workspace = torch.zeros(out_features // 64 * 16, dtype=torch.int, device=data.device)
        else:
            raise ValueError("This should not happen. Please open an issue.")
        
        super().__init__(qtype, axis, size, stride, data_repack, scale, zeropoint)

    def dequantize(self):
        # TODO: implement
        raise NotImplementedError()

    def qbits_tensor(self):
        """Convert back to a QBitsTensor

        This is required to make sure only standard packing is used when serializing.
        """
        # TODO: implement
        raise NotImplementedError()

    def __tensor_flatten__(self):
        inner_tensors = ["_data", "_scale", "_workspace"]
        meta = {
            "qtype": self._qtype.name,
            "axis": str(self._axis),
            "size": str(list(self.size())),
            "stride": str(list(self.stride())),
        }
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert len(inner_tensors) == 3
        assert len(meta) == 4
        data, scale = inner_tensors["_data"], inner_tensors["_scale"]
        # Meta should only contain strings, AST compatible except qtype
        qtype = qtypes[meta["qtype"]]
        axis = ast.literal_eval(meta["axis"])
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        return QBytesTensor(qtype, axis, size, stride, data, scale)