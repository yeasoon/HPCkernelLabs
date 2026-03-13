import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

# Test data
size = 1024
x = torch.randn(size, device='cuda')
y = torch.randn(size, device='cuda')
output = torch.empty_like(x)
add_kernel[(1,)](x, y, output, size, BLOCK_SIZE=1024)

print("Triton installation successful! Result matches:", torch.allclose(output, x + y))