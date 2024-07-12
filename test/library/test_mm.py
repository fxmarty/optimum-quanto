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

import pytest
import torch
from helpers import assert_similar, random_tensor

from optimum.quanto import AWQPackedTensor, AWQPacking


@pytest.mark.parametrize("batch_size", [1, 10, None], ids=["single", "batched", "static"])
@pytest.mark.parametrize("input_features", [32, 50])
@pytest.mark.parametrize("output_features", [48, 50, 64])
@pytest.mark.parametrize("input_dtype", [None, torch.int8], ids=["i-as-out", "i-int8"])
@pytest.mark.parametrize("weight_dtype", [torch.float8_e4m3fn, torch.int8], ids=["w-float8", "w-int8"])
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16], ids=["o-fp16", "o-bf16"])
def test_qbytes_mm(batch_size, input_features, input_dtype, weight_dtype, output_features, output_dtype, device):
    if device.type == "mps" and weight_dtype.is_floating_point:
        pytest.skip("Float8 types are not supported on MPS device")
    input_shape = (32, input_features)
    if batch_size is not None:
        input_shape = (batch_size,) + input_shape
    if input_dtype is None:
        input_dtype = output_dtype
    input = random_tensor(input_shape, dtype=input_dtype, device=device)
    weight = random_tensor((output_features, input_features), dtype=weight_dtype, device=device)
    # Use a scale small enough to prevent overflows
    scale = random_tensor((output_features,), dtype=output_dtype, device=device) / 1e3
    output = torch.ops.quanto.qbytes_mm(input, weight, scale)
    expected = torch.matmul(input.to(scale.dtype), weight.to(scale.dtype).t() * scale)
    assert_similar(expected, output)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="CUDA device >= sm80 not available",
)
@pytest.mark.parametrize("in_features, out_features", [(256, 256), (512, 256)])
@pytest.mark.parametrize("batch_size, tokens", [(4, 1), (10, 128)], ids=["gemv", "gemm"])
def test_gemm_fp16_int4(batch_size, tokens, in_features, out_features):
    """This test verifies that the GEMM operation is equivalent to torch.mm."""
    bits = 4
    group_size = 128  # Hard-coded in kernels
    device = torch.device("cuda")
    input_shape = (batch_size, tokens, in_features)
    # FIXME: does not work if inputs are negative !!??
    inputs = torch.rand(input_shape, dtype=torch.float16, device=device)
    qmax = 2**bits
    other_shape = (out_features, in_features)
    other_data = torch.randint(0, qmax, other_shape, dtype=torch.uint8, device=device)
    packed_other_data = AWQPackedTensor.pack(other_data, packing=AWQPacking.V2)._data
    # The GEMM kernel works on transposed scales
    scales_shape = (in_features // group_size, out_features)
    other_scales = torch.rand(scales_shape, dtype=torch.float16, device=device) / qmax
    # The GEMM kernel works on transposed, negated and scaled shifts
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1)
    other_shifts = torch.randint(qmin, qmax, scales_shape, dtype=torch.int8, device=device)
    # Negate and scale
    other_scaled_shifts = -other_shifts * other_scales
    # Evaluate mm outputs using the GEMM kernel
    lib_outputs = torch.ops.quanto.gemm(
        inputs,
        packed_other_data,
        other_scales,
        other_scaled_shifts,
        rows=inputs.numel() // inputs.shape[-1],
        out_cols=out_features,
        in_cols=in_features,
        bits=4,
        group_size=group_size,
    )
    # Transpose other data and reshape it to align it with transposed scales and zeros
    other_data_t = other_data.t().reshape(group_size, in_features // group_size, out_features)
    # Dequantize transposed other
    other_t = (other_data_t - other_shifts) * other_scales
    # Reshape it as expected by the matmul
    other_t = other_t.reshape(in_features, out_features)
    # Evaluate the matrix multiplication using pytorch float16 mm
    pt_outputs = torch.matmul(inputs, other_t)
    # Verify the results are similar
    assert_similar(lib_outputs, pt_outputs, rtol=5e-3)


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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="CUDA device >= sm80 not available",
)
@pytest.mark.parametrize("tokens", [1, 10, 128])
@pytest.mark.parametrize("in_features, out_features", [(256, 256), (512, 256)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_fp8_marlin(tokens, in_features, out_features, dtype):
    device = torch.device("cuda")
    input_shape = (tokens, in_features)
    inputs = torch.rand(input_shape, dtype=dtype, device=device)
    other_shape = (in_features, out_features)
    other_data = torch.rand(other_shape, dtype=dtype, device=device).to(torch.float8_e4m3fn)
    other_data_int32 = pack_fp8_as_int32(other_data)
    perm = torch.empty(0, dtype=torch.int, device=device)
    other_data_repack = torch.ops.quanto.gptq_marlin_repack(
        b_q_weight=other_data_int32, perm=perm, size_k=in_features, size_n=out_features, num_bits=8
    )
    other_scale_shape = (1, out_features)
    other_scale = torch.rand(other_scale_shape, dtype=dtype, device=device)
    workspace = torch.zeros(out_features // 64 * 16, dtype=torch.int, device=device)
    lib_outputs = torch.ops.quanto.fp8_marlin(
        a=inputs,
        b_q_weight=other_data_repack,
        b_scales=other_scale,
        workspace=workspace,
        num_bits=8,
        size_m=tokens,
        size_n=out_features,
        size_k=in_features,
    )
    # Evaluate the matrix multiplication using pytorch mm
    other = other_data.to(dtype) * other_scale
    pt_outputs = torch.matmul(inputs.to(dtype), other)
    # Verify the results are similar
    assert_similar(lib_outputs, pt_outputs)
