import argparse
import time
from typing import Tuple

import torch
import torch.nn as nn


class Model(nn.Module):
    """Reference model matching the KernelBench task structure."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)


def naive_cpu_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Naive triple-loop CPU matmul for row-major 2D tensors."""
    if A.device.type != "cpu" or B.device.type != "cpu":
        raise ValueError("naive_cpu_matmul expects CPU tensors")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("naive_cpu_matmul expects 2D tensors")
    if A.shape[1] != B.shape[0]:
        raise ValueError("inner dimensions must match")

    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    _, N = B.shape
    C = torch.zeros((M, N), dtype=A.dtype)

    for i in range(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc += float(A[i, k]) * float(B[k, j])
            C[i, j] = acc
    return C


def get_inputs(m: int, k: int, n: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    A = torch.rand((m, k), dtype=dtype)
    B = torch.rand((k, n), dtype=dtype)
    return A, B


def benchmark(fn, A: torch.Tensor, B: torch.Tensor, warmup: int, repeats: int):
    for _ in range(warmup):
        fn(A, B)

    durations = []
    out = None
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn(A, B)
        end = time.perf_counter()
        durations.append(end - start)
    return out, durations


def main():
    parser = argparse.ArgumentParser(description="Naive CPU standard matmul experiment")
    parser.add_argument("--m", type=int, default=128, help="Rows of A")
    parser.add_argument("--k", type=int, default=256, help="Cols of A / rows of B")
    parser.add_argument("--n", type=int, default=128, help="Cols of B")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per kernel")
    parser.add_argument("--repeats", type=int, default=3, help="Measured iterations per kernel")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Tensor dtype",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Torch CPU thread count for the torch.matmul baseline",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)

    dtype = getattr(torch, args.dtype)
    A, B = get_inputs(args.m, args.k, args.n, dtype)
    model = Model().cpu()

    naive_out, naive_times = benchmark(naive_cpu_matmul, A, B, args.warmup, args.repeats)
    torch_out, torch_times = benchmark(model, A, B, args.warmup, args.repeats)

    max_abs_err = (naive_out - torch_out).abs().max().item()
    flops = 2.0 * args.m * args.k * args.n
    naive_best = min(naive_times)
    torch_best = min(torch_times)

    print("=== Standard Matrix Multiplication CPU Experiment ===")
    print(f"shape: A=({args.m}, {args.k}), B=({args.k}, {args.n})")
    print(f"dtype: {args.dtype}")
    print(f"torch threads: {args.num_threads}")
    print(f"max abs error vs torch.matmul: {max_abs_err:.6e}")
    print()
    print("Naive triple-loop CPU kernel")
    print(f"  times (s): {[round(t, 6) for t in naive_times]}")
    print(f"  best time (s): {naive_best:.6f}")
    print(f"  best GFLOPS: {flops / naive_best / 1e9:.6f}")
    print()
    print("torch.matmul CPU baseline")
    print(f"  times (s): {[round(t, 6) for t in torch_times]}")
    print(f"  best time (s): {torch_best:.6f}")
    print(f"  best GFLOPS: {flops / torch_best / 1e9:.6f}")


if __name__ == "__main__":
    main()
