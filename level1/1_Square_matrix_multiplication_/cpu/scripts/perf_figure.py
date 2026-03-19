import matplotlib.pyplot as plt

block_size = [16, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096]

time_s = [
    31.2699, 22.252, 19.7984, 20.3845, 17.9413,
    16.9122, 14.5652, 13.0514, 12.623, 15.8855, 26.5968
]

gflops = [
    4.39524, 6.17647, 6.94191, 6.74234, 7.66046,
    8.12664, 9.43613, 10.5306, 10.8879, 8.65183, 5.1675
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,8))

# Time plot
ax1.plot(block_size, time_s, marker='o')
ax1.set_xscale("log", base=2)
ax1.set_xlabel("Block Size")
ax1.set_ylabel("Time (seconds)")
ax1.set_title("Block Size vs Execution Time")
ax1.grid(True)

# Performance plot
ax2.plot(block_size, gflops, marker='s')
ax2.set_xscale("log", base=2)
ax2.set_xlabel("Block Size")
ax2.set_ylabel("Performance (GFLOPS)")
ax2.set_title("Block Size vs Performance")
ax2.grid(True)

plt.tight_layout()

plt.savefig("../assets/matmul_blocksize_analysis.png", dpi=300)
# plt.show()