import torch
import triton
import triton.language as tl
import os

# --- STEP 1: Choose your Mode ---
# Option A: Debugging/Simulating (No special backend needed)
os.environ["TRITON_INTERPRET"] = "1" 

# Option B: Actual CPU Execution (Requires pytorch-labs/triton-cpu)
# os.environ["TRITON_CPU_BACKEND"] = "1" 

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Map this program instance to a specific row and column block
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute memory offsets for the tiles
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the first tiles of A and B
    a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Initialize the accumulator for the output tile
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension (tiling along the common axis)
    for k in range(0, K, BLOCK_SIZE_K):
        # Load tiles from DRAM into SRAM (or CPU Cache in this case)
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        
        # Matrix multiply the tiles and accumulate
        acc += tl.dot(a, b)

        # Advance the pointers to the next tiles along the K axis
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Final pointer to where the output tile should be stored
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptrs, acc)

def matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Define tile sizes
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']), 
        triton.cdiv(N, META['BLOCK_SIZE_N'])
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32
    )
    return c

# Testing on CPU
A = torch.randn((512, 512), device='cpu')
B = torch.randn((512, 512), device='cpu')
C_triton = matmul(A, B)
C_torch = torch.matmul(A, B)

print(f"Max difference: {torch.max(torch.abs(C_triton - C_torch))}")