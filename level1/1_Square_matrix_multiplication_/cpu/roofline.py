import numpy as np
import matplotlib.pyplot as plt

ai = np.logspace(-2,3,200)

peak = 614.4
l1_bw = 1400
l2_bw = 600
l3_bw = 200
dram_bw = 100

roof_l1 = np.minimum(peak, l1_bw * ai)
roof_l2 = np.minimum(peak, l2_bw * ai)
roof_l3 = np.minimum(peak, l3_bw * ai)
roof_mem = np.minimum(peak, dram_bw * ai)

plt.loglog(ai, roof_l1)
plt.loglog(ai, roof_l2)
plt.loglog(ai, roof_l3)
plt.loglog(ai, roof_mem)

plt.scatter([50], [10.9])

plt.xlabel("Arithmetic Intensity (FLOP/Byte)")
plt.ylabel("GFLOPS")
plt.title("Multi-level CPU Roofline")
plt.grid(True)

plt.savefig("cpu_roofline.png", dpi=300)
# plt.show()