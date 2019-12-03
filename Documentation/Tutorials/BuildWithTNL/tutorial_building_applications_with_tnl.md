\page tutorial_building_applications_with_tnl  Building Applications with TNL

## Introduction

One may find usefull to read this tutorial before any other to learn how to compile the examples. Here we explain how to build applications with TNL and we provide templates of Makefiles which can help the user when starting developing programs with TNL. Since TNL is header-only library no linker setup is required.

## Table of Contents
1. [Compilation using command-line](#command_line)
   1. [Compilation with `g++`](#command_line_gcc)
   2. [Compilation with `nvcc` for CUDA](#command_line_nvcc)
2. [Build with Makefile](#makefile)
3. [Build with Cmake](#cmake)

## Compilation using command-line  <a name="command_line"></a>

This section mainly explains how to compile with and without support of CUDA using different compilers. We start with the following simple example:

\include example-host.cpp

This short program just create new array, initiate it with values 1, 2 and 3 and print it on a console. 

### Compilation with `g++` <a name="command_line_gcc"></a>

We assume that the code above is saved in a file `example-host.cpp.` With GNU g++compiler, the program can be compiled as follows:

```
g++ -std=c++14 -I${HOME}/.local/include/tnl example-host.cpp -o example-host
```

TNL requires standard C++14 which we enforce with the first parameter `-std=c++14`. Next, we need to tell the compiler the folder with TNL headers. This is done with the flag `-I`. By default, TNL installs into `${HOME}/.local/include/tnl`. You may also replace it just with the path where you have downloaded TNL. TNL is header only library and so it does not require any instalation. Finaly, we just past the source code file `example-host.cpp` using the command-line parameter `-c`.

### Compilation with `nvcc` for CUDA <a name="command_line_nvcc"></a>

If you want to profit from the great performance of GPUs using CUDA you need to have the CUDA compiler `nvcc`. It can be obtained with the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads). We first modify our program as follows:

\include example-cuda.cpp

We need to include the header `TNL/Devices/Cuda.h` and declare the new `device_array` using a template parameter `Devices::Cuda`. For more details see [the arrays tutorial](tutorial_01_arrays.html). To compile the code above invoke the following command:

```
nvcc -I${HOME}/.local/include/tnl example-cuda.cu -o example-cuda
```

After executing the binary `example-cuda` we get error message surprisingly:

```
host_array = [ 1, 2, 3 ]
terminate called after throwing an instance of 'TNL::Exceptions::CudaSupportMissing'
  what():  CUDA support is missing, but the program called a function which needs it. Please recompile the program with CUDA support.
Aborted (core dumped)
```

The reason is that each piece of CUDA code in TNL is guarded by a macro `HAVE_CUDA`. Therefore we need to pass `-DHAVE_CUDA` to the compiler. The following command will make it:

```
nvcc -DHAVE_CUDA -I${HOME}/.local/include/tnl example-cuda.cu -o example-cuda
```

Unfortunately, `nvcc` compiler generates a lot of warnings. When used with TNL, the amount of code processed by `nvcc` is rather large and so you can get really a lot of warnings. Some of them are treated as errors by default. For this reason we recommend to add the following flags to `nvcc`:

```
-Wno-deprecated-gpu-targets --expt-relaxed-constexpr --expt-extended-lambda
```

The overall command looks as:

```
nvcc -Wno-deprecated-gpu-targets --expt-relaxed-constexpr --expt-extended-lambda -DHAVE_CUDA -I${HOME}/.local/include/tnl example-cuda.cu -o example-cuda
```

We sugest to guard the CUDA code by the macro HAVE_CUDA even in your projects. Our simple example then turns into the following:

\include example-cuda-2.h

The best way is store this code into a header file `example-cuda-2.h` for example. Include this header in files `example-cuda-2.cpp` and `example-cuda-2.cu` like this:

\include example-cuda-2.cpp

It allows you to compile with CUDA like this:

```
nvcc -Wno-deprecated-gpu-targets --expt-relaxed-constexpr --expt-extended-lambda -DHAVE_CUDA -I${HOME}/.local/include/tnl example-cuda-2.cu -o example-cuda-2
```

Or may compile it withou CUDA like this:

```
g++ -std=c++14 -I${HOME}/.local/include/tnl example-cuda-2.cpp -o example-cuda-2
```

Thus you have one code which you may easily compile with or without CUDA depending on your needs.

## Build with Makefile <a name="makefile"></a>

Larger projects needs to be managed by Makefile tool. In this section we propose a Makefile template which might help you to create more complex applications with TNL. The basic setup is stored in [Makefile.inc](../../BuildWithTNL/Makefile.inc) file:

\include Makefile.inc

In this file, you may define a name of your project (`PROJECT_NAME`), set the path to TNL headers (`TNL_HEADERS`), set the installation directory (`INSTALL_DIR`), turn on and off support of CUDA (`WITH_CUDA`), OpenMP (`WITH_OPEMP`) or debug mode (`WITH_DEBUG`). If you compile with CUDA you may set the CUDA architecture of your system.

The main [Makefile](../../BuildWithTNL/Makefile) looks as follows:

\include Makefile

If your project source codes are splitted into several subdirectories you may specify them in variable `SUBDIRS`. Next, in variables `HEADERS` and `SOURCES` you should tell all source files in the current folder. The same holds for `CUDA_SOURCES` which are all .cu files in the current folder. `TARGETS` and `CUDA_TRGETS` tell the names of binaries to be build in the current folder.

## Build with Cmake <a name="cmake"></a>


