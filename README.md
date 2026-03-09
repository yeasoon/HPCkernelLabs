# HPCkernelLabs
## checklist
- [ ] base on kernelbench level 1 part, regenerate the kernel for cpu, avalible gpu
- [ ] cpu add naive/intrinsic/assembly version and use some autogenerate tools
- [ ] gpu add naive/triton/tvm/tilelang/ptx/sass version for some specific gpu (i can get and used)
- [ ] build tools to analysis the performance, or use agent to pipline those work

## env setup
* create conda env
```bash
conda create -n hpc-kernel-labs python=3.10 -y
conda activate hpc-kernel-labs
# change the cuda version according to your env, here i use 10.2 as example
# export PATH=/home/cd_engine_group/group_common_dirs/cuda/cuda-10.2/bin:$PATH
# pip install torch==1.12.0+cu102 torchvision==0.13.0+cu102 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu102
pip install triton
pip install matplotlib
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

```
### triton-cpu
```bash
git clone https://github.com/pytorch-labs/triton-cpu.git
cd triton-cpu

# Build and install (takes 10–20 min)
pip uninstall triton -y
TRITON_BUILD_WITH_CLANG_LLD=0 pip install -e python --no-build-isolation
```
### tvm
```bash
git clone https://github.com/apache/tvm
cd tvm
git submodule update --init --recursive
conda install -c conda-forge llvmdev=16 cmake
CMAKE_ARGS="-DUSE_LLVM=llvm-config -DUSE_OPENMP=ON" pip install -e .

```

## perf tools
https://gitlab.com/hpctoolkit/hpctoolkit.git

| Tool          | Precision | Overhead | Best For                                                      |
|---------------|-----------|----------|---------------------------------------------------------------|
| `time.time()` | Low       | None     | Quick wall-clock checks and manual timing.                   |
| `PyInstrument`| Medium    | Low      | Identifying which Python function or block is slow.          |
| `cProfile`    | High      | High     | Deterministic profiling; counting every single function call.|
| `perf (Linux)`| Extreme   | Very Low | Deep analysis of C++/CUDA kernels and hardware cache misses. |

| Tool              | Best Use Case              | HPC Benefit                                                     |
|-------------------|----------------------------|-----------------------------------------------------------------|
| `lstopo`          | Visualizing the full tree  | Identifying shared caches for MPI/OpenMP pinning.              |
| `likwid-topology` | Thread-to-core mapping     | Precise control over hardware thread placement.                |
| `numactl`         | Memory affinity            | Preventing "remote" memory access across sockets.              |
| `nvidia-smi`      | GPU Inventory              | Monitoring SM utilization and memory thermals.                 |
| `cpuid`           | Feature verification       | Checking for specific SIMD sets (e.g., AVX-512).               |

```
lscpu
lstopo --of png > topology.png

nvidia-smi topo -m
cuda-10.1/extras/demo_suite/deviceQuery 
```