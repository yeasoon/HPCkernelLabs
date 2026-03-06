import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

def naive_cpu_matmul(A, B):
    """
    Standard Triple-Loop Matrix Multiplication (C = A * B)
    A: (M, K)
    B: (K, N)
    C: (M, N)
    """
    M, K = A.shape
    K_alt, N = B.shape
    
    # Initialize output matrix with zeros
    C = torch.zeros((M, N))

    # Triple Nested Loop
    for i in range(M):          # Loop over rows of A
        for j in range(N):      # Loop over columns of B
            sum_val = 0.0
            for k in range(K):  # Dot product of row A[i] and col B[j]
                sum_val += A[i, k] * B[k, j]
            C[i, j] = sum_val
    return C


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # The record_function label will show up in your profile report
        with record_function("model_inference"):
            return torch.matmul(A, B)

N = 2048 * 2

def get_inputs(device):
    A = torch.rand(N, N).to(device)
    B = torch.rand(N, N).to(device)
    return [A, B]

def profile_model(model, inputs):
    print("--- Starting Profile ---")
    
    # Use both CPU and CUDA activities
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    
    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        with record_function("total_execution"):
            model(*inputs)

    # Print the results sorted by CUDA time
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Optional: Export for Chrome Trace (view at chrome://tracing)
    # prof.export_chrome_trace("trace.json")

def main():
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Model().to(device)
    inputs = get_inputs(device)
    
    # Warm up (to avoid initialization overhead in profile)
    model(*inputs)
    
    # Run Profile
    profile_model(model, inputs)

if __name__ == "__main__":    
    main()