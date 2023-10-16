# Copyright (c) Meta Platforms, Inc. and affiliates.
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

import argparse
import json
from typing import Callable, Dict, Tuple

import torch
import torch.utils.benchmark as benchmark
from layers import LayerFactory, LayerType
from utils import get_layer_set, reset_peak_memory_stats


def run_layer_benchmark(
    num_repeats: int,
    forward_only: bool = False,
    create_layer: Callable = LayerFactory.create,
    **kwargs,
) -> Tuple[float, Dict[str, int]]:
    """Benchmarks a single layer for runtime and CUDA memory (if applicable).

    Args:
        num_repeats: how many times to repeat the forward(/backward) pass
        forward_only: whether to skip the backward pass
        create_layer: function for creating the layer, takes **kwargs

    Returns: a tuple consisting of
        Runtime as a float, and
        Memory statistics as a dict
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        assert reset_peak_memory_stats(device).cur_mem == 0

    # setup layer
    layer_fun = create_layer(**kwargs)

    if forward_only:
        layer_fun.module.eval()
        benchmark_fun = layer_fun.forward_only
    else:
        layer_fun.module.train()
        benchmark_fun = layer_fun.forward_backward

    # move layer to device and get memory statistics
    memory_stats = layer_fun.to(device=device)
    assert sum(v for _, v in memory_stats.items()) == torch.cuda.memory_allocated(
        device
    )

    # benchmark.Timer performs its own warmups
    try:
        timer = benchmark.Timer(
            stmt="benchmark_fun()",
            globals={"benchmark_fun": benchmark_fun},
            num_threads=1,
        )
        runtime = timer.timeit(num_repeats).mean
    except RuntimeError:
        runtime = float("nan")

    # get max memory allocated and reset memory statistics
    memory_stats["max_memory"] = reset_peak_memory_stats(device).prev_max_mem

    return runtime, memory_stats


def main(args) -> None:

    with open(args.config_file) as config_file:
        config = json.load(config_file)

    runtime, memory_stats = run_layer_benchmark(
        num_repeats=args.num_repeats,
        forward_only=args.forward_only,
        layer_name=args.layer,
        batch_size=args.batch_size,
        random_seed=args.random_seed,
        **config[get_layer_set(args.layer)],
    )
    print(f"Runtime (seconds): {runtime}")
    print(f"Memory statistics (bytes): {memory_stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "layer",
        type=str,
        choices=[v for k, v in LayerType.__dict__.items() if not k.startswith("__")],
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument(
        "--num_repeats",
        default=20,
        type=int,
        help="number of forward/backward passes to run",
    )
    parser.add_argument(
        "--forward_only", action="store_true", help="only run forward passes"
    )
    parser.add_argument("--random_seed", default=0, type=int)
    parser.add_argument(
        "-c",
        "--config_file",
        default="config.json",
        type=str,
        help="path to config file with settings for each layer",
    )
    args = parser.parse_args()
    main(args)
