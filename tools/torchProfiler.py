import torch
from torch.profiler import profile, record_function, ProfilerActivity
import functools

class TorchProfiler:
    def __init__(self, name="Profile", enabled=True, use_cuda=True):
        self.name = name
        self.enabled = enabled
        self.activities = [ProfilerActivity.CPU]
        if use_cuda and torch.cuda.is_available():
            self.activities.append(ProfilerActivity.CUDA)

    def __enter__(self):
        if not self.enabled:
            return self
        
        self.prof = profile(
            activities=self.activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        self.prof.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        
        self.prof.__exit__(exc_type, exc_val, exc_tb)
        print(f"\n{'='*20} {self.name} {'='*20}")
        print(self.prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        # Optional: Save trace for Chrome/Perfetto
        # self.prof.export_chrome_trace(f"{self.name.replace(' ', '_')}_trace.json")

    def __call__(self, func):
        """Allows the class to be used as a decorator (@TorchProfiler())"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

'''
usage example:
@TorchProfiler(name="Model Forward Pass")
def train_step(model, inputs):
    return model(*inputs)

with TorchProfiler(name="Matrix Mult Latency"):
        output = model(*inputs)
'''