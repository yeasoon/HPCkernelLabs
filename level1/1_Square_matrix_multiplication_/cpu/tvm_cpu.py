import tvm
from tvm.script import tir as T
def matmul_test():
    import numpy as np
    import time

    N = 4096

    @T.prim_func
    def matmul(A: T.Buffer((N, N), "float32"),
            B: T.Buffer((N, N), "float32"),
            C: T.Buffer((N, N), "float32")):

        for i in T.parallel(N):
            for j in range(N):
                C[i, j] = 0.0
                for k in range(N):
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]


    mod = tvm.IRModule({"matmul": matmul})

    rt = tvm.build(mod, target="llvm")

    dev = tvm.cpu()

    A_np = np.random.rand(N, N).astype("float32")
    B_np = np.random.rand(N, N).astype("float32")
    C_np = np.zeros((N, N), dtype="float32")

    A = tvm.runtime.tensor(A_np, dev)
    B = tvm.runtime.tensor(B_np, dev)
    C = tvm.runtime.empty((N, N), "float32", dev)

    # warmup
    rt(A, B, C)

    # timing
    iters = 10
    start = time.time()
    for _ in range(iters):
        rt(A, B, C)
    end = time.time()

    elapsed = (end - start) / iters

    flops = 2 * N * N * N
    gflops = flops / elapsed / 1e9

    print("Time per run:", elapsed, "seconds")
    print("Performance:", gflops, "GFLOPS")

    # correctness check
    C_ref = A_np @ B_np
    print("Correct:", np.allclose(C.numpy(), C_ref, atol=1e-3))

    # dump assembly
    # print("\nGenerated assembly snippet:\n")
    # print(rt.inspect_source("asm")[:800])
def matmul_tiled_test():
    import numpy as np
    import time

    N = 4096
    BS = 1024   # tile size

    @T.prim_func
    def matmul_tiled(A: T.Buffer((N, N), "float32"),
                    B: T.Buffer((N, N), "float32"),
                    C: T.Buffer((N, N), "float32")):

        for ii in T.parallel(N // BS):
            for jj in range(N // BS):

                for i in range(BS):
                    for j in range(BS):
                        C[ii*BS+i, jj*BS+j] = 0.0

                for kk in range(N // BS):
                    for i in range(BS):
                        for j in range(BS):
                            for k in range(BS):
                                C[ii*BS+i, jj*BS+j] += (
                                    A[ii*BS+i, kk*BS+k] *
                                    B[kk*BS+k, jj*BS+j]
                                )

    mod = tvm.IRModule({"matmul_tiled": matmul_tiled})

    rt = tvm.build(mod, target="llvm")

    dev = tvm.cpu()

    A_np = np.random.rand(N, N).astype("float32")
    B_np = np.random.rand(N, N).astype("float32")

    A = tvm.runtime.tensor(A_np, dev)
    B = tvm.runtime.tensor(B_np, dev)
    C = tvm.runtime.empty((N, N), "float32", dev)

    # warmup
    rt(A, B, C)

    # timing
    iters = 5
    start = time.time()
    for _ in range(iters):
        rt(A, B, C)
    end = time.time()

    t = (end - start) / iters
    flops = 2 * N * N * N
    gflops = flops / t / 1e9

    print("time:", t)
    print("GFLOPS:", gflops)

    # correctness
    print("correct:", np.allclose(C.numpy(), A_np @ B_np, atol=1e-3))

    # show assembly snippet
    # print("\nASM preview\n")
    # print(rt.inspect_source("asm")[:800])
def vec_add_test():
    @T.prim_func
    def vec_add(A: T.Buffer((1024,), "float32"),
                B: T.Buffer((1024,), "float32"),
                C: T.Buffer((1024,), "float32")):

        for i in range(1024):
            C[i] = A[i] + B[i]

    mod = tvm.IRModule({"vec_add": vec_add})

    rt = tvm.build(mod, target="c")
    print(rt.inspect_source())
matmul_test()
matmul_tiled_test()
# vec_add_test()