# Vectors tutorial

## Introduction

This tutorial introduces vectors in TNL. `Vector`, in addition to `Array`, offers also basic operations from linear algebra. Methods implemented in arrays and vectors are particularly usefull for GPU programming. From this point of view, the reader will learn how to easily allocate memory on GPU, transfer data between GPU and CPU but also, how to initialise data allocated on GPU and perform parallel reduction and vector operations without writting low-level CUDA kernels. In addition, the resulting code is hardware platform independent, so it can be ran on CPU without any changes.

# Table of Contents
1. [Vectors](#vectors)
2. [Static vectors](#static_vectors)

## Vectors <a name="vectors"></a>

`Vector` is, similar to `Array`, templated class defined in namespace `TNL::Containers` having three template parameters:

* `Real` is type of data to be stored in the vector
* `Device` is the device where the vector is allocated. Currently it can be either `Devices::Host` for CPU or `Devices::Cuda` for GPU supporting CUDA.
* `Index` is the type to be used for indexing the vector elements.

`Vector`, unlike `Array`, requires that the `Real` type is numeric or a type for which basic algebraic operations are defined. What kind of algebraic operations is required depends on what vector operations the user will call. `Vector` is derived from `Array` so it inherits all its methods. In the same way the `Array` has its counterpart `ArraView`, `Vector` has `VectorView` which is derived from `ArrayView`. We refer to to [Arrays tutorial](../Arrays/index.html) for more details.

### Vector expressions

Vector expressions in TNL are processed by the [Expression Templates](https://en.wikipedia.org/wiki/Expression_templates). It makes algebraic operations with vectors easy to do and very efficient at the same time. In some cases, one get even more efficient code compared to [Blas](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) and [cuBlas](https://developer.nvidia.com/cublas). See the following example to learn how simple it is.

\include Expressions.cpp

Output is:

\include Expressions.out


## Static vectors <a name="static_vectors"></a>
