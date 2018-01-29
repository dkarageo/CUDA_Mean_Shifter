# **CUDA Mean Shifter**

## An implementation of Mean Shift algorithm on CUDA, with a variant that utilizes shared memory and one that does not.


Developed by *Dimitrios Karageorgiou*,\
during the course *Parallel And Distributed Systems*,\
*Aristotle University Of Thessaloniki, Greece,*\
*2017-2018.*

Provided in this repo, are two variants of Mean Shift implemented on CUDA. Also there is a CPU implementation of Mean Shift, helpful for comparing gpu version against a CPU one.

### **How to compile:**
```
make
```
This way, a demo that includes all three variants is produced.

If selective compiling is needed the following targets are available:
```
make cpu_only
```
which produces a demo only with CPU variant,
```
make gpu_only
```
which produces a demo only with GPU variant that utilizes shared memory,
```
make gpu_with_nosm
```
which produces a demo containing both GPU variants.

In order to successfully compile, NVIDIA CUDA Driver should be installed, since it utilizes `nvcc`. Also a host C compiler with at least C99 support should be present.

### **How to run:**

```
make run
```
or
```
make runp data=<path_to_datafile> h=<mean_shift_scalar>
```

If selective running is needed, the following targets are available:
```
make run_cpu
make run_gpu
make run_gpu_all
```
or
```
make runp_cpu data=<path_to_datafile> h=<mean_shift_scalar>
make runp_gpu data=<path_to_datafile> h=<mean_shift_scalar>
make runp_gpu_all data=<path_to_datafile> h=<mean_shift_scalar>
```
