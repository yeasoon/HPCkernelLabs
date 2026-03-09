## ncu need sudo to run, so we need to add sudo before ncu command

```bash
sudo cp ncu-autho.conf  /etc/modprobe.d/
sudo update-initramfs -u
sudo reboot
```
### 1080ti not support ncu, so we need to use nvprof instead

```bash
nvprof ./matrixMul
nvprof --print-gpu-trace ./matrixMul
```