#!/usr/bin/env python3
# Copyright 2026 Xiaomi Corporation
#
# See ../../../../LICENSE for clarification regarding multiple authors
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
"""Zipformer streaming-state batching helpers.

The state schema mirrors the recipe's established streaming_decode.py helpers.
Each input item represents one stream with batch size 1.
"""

from __future__ import annotations

from typing import Any, List


def stack_states(state_list: List[List[Any]]) -> List[Any]:
    import torch

    if not state_list:
        raise ValueError("cannot stack an empty streaming-state list")
    state_count = len(state_list[0])
    if state_count < 2 or (state_count - 2) % 6 != 0:
        raise ValueError("unsupported Zipformer state count: {}".format(state_count))
    if any(len(states) != state_count for states in state_list):
        raise ValueError("all streams must use the same Zipformer state schema")

    batch_size = len(state_list)
    layer_count = (state_count - 2) // 6
    batch_states = []
    for layer in range(layer_count):
        offset = layer * 6
        for state_offset, batch_dim in ((0, 1), (1, 1), (2, 1), (3, 1)):
            batch_states.append(
                torch.cat(
                    [state_list[i][offset + state_offset] for i in range(batch_size)],
                    dim=batch_dim,
                )
            )
        for state_offset in (4, 5):
            batch_states.append(
                torch.cat(
                    [state_list[i][offset + state_offset] for i in range(batch_size)],
                    dim=0,
                )
            )
    batch_states.append(torch.cat([states[-2] for states in state_list], dim=0))
    batch_states.append(torch.cat([states[-1] for states in state_list], dim=0))
    return batch_states


def unstack_states(batch_states: List[Any]) -> List[List[Any]]:
    if len(batch_states) < 2 or (len(batch_states) - 2) % 6 != 0:
        raise ValueError(
            "unsupported batched Zipformer state count: {}".format(len(batch_states))
        )
    batch_size = int(batch_states[-1].shape[0])
    if batch_size <= 0:
        raise ValueError("batched Zipformer states have an empty batch")
    layer_count = (len(batch_states) - 2) // 6
    result: List[List[Any]] = [[] for _ in range(batch_size)]
    for layer in range(layer_count):
        offset = layer * 6
        chunks = [
            batch_states[offset].chunk(batch_size, dim=1),
            batch_states[offset + 1].chunk(batch_size, dim=1),
            batch_states[offset + 2].chunk(batch_size, dim=1),
            batch_states[offset + 3].chunk(batch_size, dim=1),
            batch_states[offset + 4].chunk(batch_size, dim=0),
            batch_states[offset + 5].chunk(batch_size, dim=0),
        ]
        if any(len(values) != batch_size for values in chunks):
            raise ValueError("Zipformer state batch dimensions are inconsistent")
        for index in range(batch_size):
            result[index].extend(values[index] for values in chunks)
    embed_chunks = batch_states[-2].chunk(batch_size, dim=0)
    length_chunks = batch_states[-1].chunk(batch_size, dim=0)
    if len(embed_chunks) != batch_size or len(length_chunks) != batch_size:
        raise ValueError("Zipformer trailing state batch dimensions are inconsistent")
    for index in range(batch_size):
        result[index].extend((embed_chunks[index], length_chunks[index]))
    return result
