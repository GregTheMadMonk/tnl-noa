# Vectors tutorial

## Introduction

This tutorial introduces vectors in TNL. ```Vector```, in addition to ```Array```, offers also basic operations from linear algebra. Methods implemented in arrays and vectors are particularly usefull for GPU programming. From this point of view, the reader will learn how to easily allocate memory on GPU, transfer data between GPU and CPU but also, how to initialise data allocated on GPU and perform parallel reduction and vector operations without writting low-level CUDA kernels. In addition, the resulting code is hardware platform independent, so it can be ran on CPU without any changes.

## Vectors

```Vector``` is, similar to ```Array``` templated class defined in namespace ```TNL::Containers``` having three template parameters:

* ```Value``` is type of data to be stored in the array
* ```Device``` is the device wheer the array is allocated. Currently it can be either ```Devices::Host``` for CPU or ```Devices::Cuda``` for GPU supporting CUDA.
* ```Index``` is the type to be used for indexing the array elements.

