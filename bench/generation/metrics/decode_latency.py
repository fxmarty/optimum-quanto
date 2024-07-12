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

import gc
import time

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import GenerationConfig
from torch.profiler import ProfilerActivity, profile


def decode_latency(model, tokenizer, device, batch_size=1, nb_tokens=512, iterations=10):
    def synchronize(device):
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        else:
            torch.cpu.synchronize()

    def timing_event(device):
        if device.type == "cuda":
            return torch.cuda.Event(enable_timing=True)
        elif device.type == "mps":
            return torch.mps.Event(enable_timing=True)

        class CPUEvent:
            def __init__(self):
                self.time = None

            def record(self):
                self.time = time.time()

            def elapsed_time(self, other):
                assert self.time is not None
                assert other.time is not None
                return (other.time - self.time) * 1000

        return CPUEvent()

    generation_config = GenerationConfig(
        max_new_tokens=nb_tokens,
        min_new_tokens=nb_tokens,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        num_beams=1,
        do_sample=False,
        eos_token_id=None,  # This is required for min_new_tokens to actually have an effect.
    )
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = None  # greedy_search falls back on this eos_token_id that we need to set to None as well for min_new_tokens to have an effect.

    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    memory = get_device_memory(device)
    if memory is not None:
        print(f"Device memory: {memory / (2 ** 30):.4f} GB")

    latencies = []
    input_ids = torch.randint(1, model.config.vocab_size - 1, size=(batch_size, 1)).to(device)
    masks = torch.ones(batch_size, 1, dtype=torch.int32).to(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(iterations):
            # start_event = timing_event(device)
            # end_event = timing_event(device)
            # synchronize(device)
            # start_event.record()

            # _ = model.generate(input_ids, attention_mask=masks, generation_config=generation_config)
            # end_event.record()
            # synchronize(device)

            # latency_ms = start_event.elapsed_time(end_event)
            _ = model.generate(input_ids, attention_mask=masks, generation_config=generation_config)

            latencies.append(5)
    prof.export_chrome_trace("trace.json")

    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated()
        print(f"Peak memory during benchmark: {peak_memory / (2 ** 30):.4f} GB")

    mean_latency = np.mean(latencies) / nb_tokens
    print(f"Average decode latency per token: {mean_latency} ms")
    return mean_latency


def get_device_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated()
    elif device.type == "mps":
        torch.mps.empty_cache()
        return torch.mps.current_allocated_memory()
    return None
