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

from typing import List, Tuple

import torch

from ..qbytes import QBytesTensor


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

        out_features, in_features = data.shape
        self._workspace = torch.zeros(out_features // 64 * 16, dtype=torch.int, device=data.device)

        scale = scale.repeat(1, out_features).to(data.device)
        data_int32 = pack_fp8_as_int32(data.T)
        perm = torch.empty(0, dtype=torch.int, device=data.device)

        data_repack = torch.ops.quanto.gptq_marlin_repack(
            b_q_weight=data_int32, perm=perm, size_k=in_features, size_n=out_features, num_bits=8
        )
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
        # TODO implement
        raise NotImplementedError()

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        # TODO implement
        raise NotImplementedError()
