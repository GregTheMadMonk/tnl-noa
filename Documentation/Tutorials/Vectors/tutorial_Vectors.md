\page tutorial_Vectors  Vectors tutorial

[TOC]

## Introduction

This tutorial introduces vectors in TNL. `Vector`, in addition to `Array`, offers also basic operations from linear algebra. The reader will mainly learn how to do Blas level 1 operations in TNL. Thanks to the design of TNL, it is easier to implement, hardware architecture transparent and in some cases even faster then [Blas](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) or [cuBlas](https://developer.nvidia.com/cublas) implementation.

## Vectors

`Vector` is, similar to `Array`, templated class defined in namespace `TNL::Containers` having three template parameters:

* `Real` is type of data to be stored in the vector
* `Device` is the device where the vector is allocated. Currently it can be either `Devices::Host` for CPU or `Devices::Cuda` for GPU supporting CUDA.
* `Index` is the type to be used for indexing the vector elements.

`Vector`, unlike `Array`, requires that the `Real` type is numeric or a type for which basic algebraic operations are defined. What kind of algebraic operations is required depends on what vector operations the user will call. `Vector` is derived from `Array` so it inherits all its methods. In the same way the `Array` has its counterpart `ArraView`, `Vector` has `VectorView` which is derived from `ArrayView`. We refer to to [Arrays tutorial](../../Arrays/html/index.html) for more details.

### Horizontal operations

By *horizontal* operations we mean vector expressions where we have one or more vectors as an input and a vector as an output. In TNL, this kind of operations is performed by the [Expression Templates](https://en.wikipedia.org/wiki/Expression_templates). It makes algebraic operations with vectors easy to do and very efficient at the same time. In some cases, one get even more efficient code compared to [Blas](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) and [cuBlas](https://developer.nvidia.com/cublas). See the following example.

\include Expressions.cpp

Output is:

\include Expressions.out

Vector expressions work only with `VectorView` not with `Vector`. The expression is evaluated on the same device where the vectors are allocated, this is done automatically. One cannot, however, mix vectors from different devices in one expression. Vector expression may contain any common function like `min`, `max`, `abs`, `sin`, `cos`, `exp`, `log`, `sqrt`, `pow` etc.

### Vertical operations

By *vertical operations* we mean (parallel) reduction based operations where we have one vector expressions as an input and one value as an output. For example computing scalar product, vector norm or finding minimum or maximum of vector elements is based on reduction. See the following example.

\include Reduction.cpp

Output is:

\include Reduction.out

## Static vectors

Static vectors are derived from static arrays and so they are allocated on the stack and can be created in CUDA kernels as well. Their size is fixed as well and it is given by a template parameter. Static vector is a templated class defined in namespace `TNL::Containers` having two template parameters:

* `Size` is the array size.
* `Real` is type of numbers stored in the array.

The interface of StaticVectors is smillar to Vector. Probably the most important methods are those related with static vector expressions which are handled by expression templates. They make the use of static vectors simpel and efficient at the same time. See the following simple demonstration:

\include StaticVectorExample.cpp

The output looks as:

\include StaticVectorExample.out

## Distributed vectors

TODO
