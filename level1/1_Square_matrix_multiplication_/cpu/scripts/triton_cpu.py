import torch
import triton
import triton.language as tl

# ── Test 1: does a simple vector add compile? ──────────────────────────
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x    = tl.load(x_ptr + offs, mask=mask)
    y    = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

N = 1024
x = torch.randn(N)
y = torch.randn(N)
z = torch.zeros(N)
add_kernel[(N // 32,)](x, y, z, N, BLOCK=32)
print("add:", "✅" if torch.allclose(z, x+y) else "❌")

# ── Test 2: does tl.sum over 1D work? ─────────────────────────────────
@triton.jit
def sum_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x    = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s    = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, s)

x   = torch.randn(32)
out = torch.zeros(1)
sum_kernel[(1,)](x, out, 32, BLOCK=32)
print("sum:", "✅" if abs(out[0].item() - x.sum().item()) < 1e-4 else "❌")

# ── Test 3: does 2D load + store work? ────────────────────────────────
@triton.jit
def copy2d_kernel(src_ptr, dst_ptr, M, N,
                  stride_m, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    rm    = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn    = tl.arange(0, BLOCK_N)
    mask  = (rm[:, None] < M) & (rn[None, :] < N)
    vals  = tl.load(src_ptr + rm[:, None] * stride_m + rn[None, :], mask=mask)
    tl.store(dst_ptr + rm[:, None] * stride_m + rn[None, :], vals, mask=mask)

src = torch.randn(64, 64)
dst = torch.zeros(64, 64)
copy2d_kernel[(4,)](src, dst, 64, 64, src.stride(0), BLOCK_M=16, BLOCK_N=64)
print("copy2d:", "✅" if torch.allclose(src, dst) else "❌")

# ── Test 4: does elementwise 2D multiply + store work? ────────────────
@triton.jit
def mul2d_kernel(a_ptr, b_ptr, c_ptr, M, N,
                 stride_m, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    rm    = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn    = tl.arange(0, BLOCK_N)
    mask  = (rm[:, None] < M) & (rn[None, :] < N)
    a     = tl.load(a_ptr + rm[:, None] * stride_m + rn[None, :], mask=mask)
    b     = tl.load(b_ptr + rm[:, None] * stride_m + rn[None, :], mask=mask)
    tl.store(c_ptr + rm[:, None] * stride_m + rn[None, :], a * b, mask=mask)

a = torch.randn(64, 64)
b = torch.randn(64, 64)
c = torch.zeros(64, 64)
mul2d_kernel[(4,)](a, b, c, 64, 64, a.stride(0), BLOCK_M=16, BLOCK_N=64)
print("mul2d:", "✅" if torch.allclose(c, a*b) else "❌")

# ── Test 5: does tl.sum over axis=1 on 2D work? ───────────────────────
@triton.jit
def rowsum_kernel(x_ptr, out_ptr, M, N,
                  stride_m, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    rm    = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn    = tl.arange(0, BLOCK_N)
    mask  = (rm[:, None] < M) & (rn[None, :] < N)
    x     = tl.load(x_ptr + rm[:, None] * stride_m + rn[None, :],
                    mask=mask, other=0.0)
    s     = tl.sum(x, axis=1)   # (BLOCK_M,)
    tl.store(out_ptr + rm, s, mask=rm < M)

x   = torch.randn(64, 64)
out = torch.zeros(64)
rowsum_kernel[(4,)](x, out, 64, 64, x.stride(0), BLOCK_M=16, BLOCK_N=64)
print("rowsum:", "✅" if torch.allclose(out, x.sum(dim=1)) else "❌")