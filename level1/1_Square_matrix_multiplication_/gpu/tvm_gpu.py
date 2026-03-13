import tvm
from tvm.script import tir as T

def matmul_cuda_test():
    import numpy as np

    N = 4096

    # Tunable launch shape
    TILE_X = 16
    TILE_Y = 16

    @T.prim_func
    def matmul_cuda(
        A: T.Buffer((N, N), "float32"),
        B: T.Buffer((N, N), "float32"),
        C: T.Buffer((N, N), "float32"),
    ):
        T.func_attr({"global_symbol": "matmul_cuda", "tir.noalias": True})

        for by in T.thread_binding((N + TILE_Y - 1) // TILE_Y, thread="blockIdx.y"):
            for bx in T.thread_binding((N + TILE_X - 1) // TILE_X, thread="blockIdx.x"):
                for ty in T.thread_binding(TILE_Y, thread="threadIdx.y"):
                    for tx in T.thread_binding(TILE_X, thread="threadIdx.x"):
                        i = by * TILE_Y + ty
                        j = bx * TILE_X + tx

                        if i < N and j < N:
                            C[i, j] = T.float32(0)
                            for k in range(N):
                                C[i, j] = C[i, j] + A[i, k] * B[k, j]

    mod = tvm.IRModule({"matmul_cuda": matmul_cuda})

    # Build for NVIDIA GPU
    rt = tvm.build(mod, target="cuda")

    dev = tvm.cuda(0)
    if not dev.exist:
        raise RuntimeError("CUDA device 0 is not available")

    A_np = np.random.rand(N, N).astype("float32")
    B_np = np.random.rand(N, N).astype("float32")

    A = tvm.nd.array(A_np, device=dev)
    B = tvm.nd.array(B_np, device=dev)
    C = tvm.nd.empty((N, N), "float32", device=dev)

    # Warmup
    rt(A, B, C)

    # TVM-native timing
    timer = rt.time_evaluator(rt.entry_name, dev, number=1, repeat=10)
    prof = timer(A, B, C)

    elapsed = prof.mean
    flops = 2 * N * N * N
    gflops = flops / elapsed / 1e9

    print("Time per run:", elapsed, "seconds")
    print("Performance:", gflops, "GFLOPS")

    # correctness check
    C_ref = A_np @ B_np
    print("Correct:", np.allclose(C.numpy(), C_ref, atol=1e-2, rtol=1e-2))

    # optional
    # print(rt.imported_modules[0].get_source())
matmul_cuda_test()
