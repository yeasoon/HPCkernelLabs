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


def naive_gpu_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Naive triple-loop matmul that keeps tensors on CUDA.

    This is intentionally a literal baseline: every multiply-add is expressed as a
    small CUDA tensor operation, so it is only practical for very small matrices.
    """
    if A.device.type != "cuda" or B.device.type != "cuda":
        raise ValueError("naive_gpu_matmul expects CUDA tensors")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("naive_gpu_matmul expects 2D tensors")
    if A.shape[1] != B.shape[0]:
        raise ValueError("inner dimensions must match")

    A = A.contiguous()
    B = B.contiguous()

    m, k = A.shape
    _, n = B.shape
    C = torch.zeros((m, n), dtype=A.dtype, device=A.device)

    for i in range(m):
        for j in range(n):
            acc = torch.zeros((), dtype=A.dtype, device=A.device)
            for kk in range(k):
                acc = acc + A[i, kk] * B[kk, j]
            C[i, j] = acc
    return C


def get_inputs(m: int, k: int, n: int, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    A = torch.rand((m, k), dtype=dtype, device=device)
    B = torch.rand((k, n), dtype=dtype, device=device)
    return A, B


def benchmark_cuda(fn, A: torch.Tensor, B: torch.Tensor, warmup: int, repeats: int):
    for _ in range(warmup):
        fn(A, B)
    torch.cuda.synchronize(A.device)

    durations = []
    out = None
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn(A, B)
        torch.cuda.synchronize(A.device)
        end = time.perf_counter()
        durations.append(end - start)
    return out, durations


def main():
    parser = argparse.ArgumentParser(description="Naive GPU standard matmul experiment")
    parser.add_argument("--m", type=int, default=16, help="Rows of A")
    parser.add_argument("--k", type=int, default=32, help="Cols of A / rows of B")
    parser.add_argument("--n", type=int, default=16, help="Cols of B")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per kernel")
    parser.add_argument("--repeats", type=int, default=3, help="Measured iterations per kernel")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "float64"],
        default="float32",
        help="Tensor dtype",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device string, for example 'cuda' or 'cuda:0'",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this environment")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dtype = getattr(torch, args.dtype)
    A, B = get_inputs(args.m, args.k, args.n, dtype, device)
    model = Model().to(device)

    naive_out, naive_times = benchmark_cuda(naive_gpu_matmul, A, B, args.warmup, args.repeats)
    torch_out, torch_times = benchmark_cuda(model, A, B, args.warmup, args.repeats)

    max_abs_err = (naive_out - torch_out).abs().max().item()
    flops = 2.0 * args.m * args.k * args.n
    naive_best = min(naive_times)
    torch_best = min(torch_times)

    print("=== Standard Matrix Multiplication GPU Experiment ===")
    print(f"device: {device}")
    print(f"shape: A=({args.m}, {args.k}), B=({args.k}, {args.n})")
    print(f"dtype: {args.dtype}")
    print(f"max abs error vs torch.matmul: {max_abs_err:.6e}")
    print("note: naive_gpu_matmul is a literal triple-loop CUDA baseline and is only practical for tiny matrices")
    print()
    print("Naive triple-loop GPU kernel")
    print(f"  times (s): {[round(t, 6) for t in naive_times]}")
    print(f"  best time (s): {naive_best:.6f}")
    print(f"  best GFLOPS: {flops / naive_best / 1e9:.6f}")
    print()
    print("torch.matmul GPU baseline")
    print(f"  times (s): {[round(t, 6) for t in torch_times]}")
    print(f"  best time (s): {torch_best:.6f}")
    print(f"  best GFLOPS: {flops / torch_best / 1e9:.6f}")


if __name__ == "__main__":
    main()
