# **CUDA Mean Shifter**

## An implementation of Mean Shift algorithm on CUDA, with a variant that utilizes shared memory and one that does not.


Developed by *Dimitrios Karageorgiou*,\
during the course *Parallel And Distributed Systems*,\
*Aristotle University Of Thessaloniki, Greece,*\
*2017-2018.*

Provided in this repository, are two variants of *Mean Shift* implemented on *CUDA*. Also there is a CPU implementation of *Mean Shift*, helpful for comparing GPU version against a CPU one.

The main goal of the project is to provide an efficient implementation of *Mean Shift* on *CUDA*. The secondary goal though, is to measure the benefits achieved by utilizing *Shared Memory* feature. With this in mind, the provided implementations can be used as standalone ones for use in projects which need an actual *Mean Shift* implementation, but also used with the provided demo and possibly the provided datasets for benchmarking.

### **How to compile:**
```
make
```
This way, a demo that includes all three variants is produced.

If compiling of a demo that only includes a subset of provided implementations is needed, the following *Makefile* targets are available:
```
make cpu_only
```
which produces a demo only containing the CPU variant,
```
make gpu_only
```
which produces a demo only containing GPU variant that utilizes shared memory,
```
make gpu_with_nosm
```
which produces a demo containing both GPU variants, but not the CPU one.

In order to successfully compile demos that contain at least one GPU implementation, NVIDIA CUDA Driver should be installed, since it utilizes `nvcc`. Also, in any case a host C compiler with at least C99 support should be present.

### **How to run:**

```
make run
```
or
```
make runp data=<path_to_datafile> h=<mean_shift_scalar>
```
where *path_to_datafile* argument allows specifying a dataset file to be used and *mean_shift_scalar* the value to be used as a scalar on internal gaussian kernel.

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
**WARNING: All the above run targets, require the equivalent compile target to be manually run first, otherwise they will fail.**

### **Licensing:**

This project is licensed under GNU GPL v3.0 license.
