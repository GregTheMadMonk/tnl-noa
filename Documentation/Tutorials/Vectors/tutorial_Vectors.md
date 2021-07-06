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

\includelineno Expressions.cpp

Output is:

\include Expressions.out

Vector expressions work only with `VectorView` not with `Vector`. The expression is evaluated on the same device where the vectors are allocated, this is done automatically. One cannot, however, mix vectors from different devices in one expression. Vector expression may contain any common function like the following:

| Function          | Meaning                                                     |
|-------------------|-------------------------------------------------------------|
| \ref TNL::min     | Minimas of input vector expressions elements.               |
| \ref TNL::max     | Maximas of input vector expressions elements.               |
| \ref TNL::abs     | Absolute values of input vector expression elements.        |
| \ref TNL::sin     | Sine of input vector expression elements.                   |
| \ref TNL::cos     | Cosine of input vector expression elements.                 |
| \ref TNL::tan     | Tangent of input vector expression elements.                |
| \ref TNL::asin    | Arc sine of input vector expression elements.               |
| \ref TNL::acos    | Arc cosine of input vector expression elements.             |
| \ref TNL::atan    | Arc tangent of input vector expression elements.            |
| \ref TNL::sinh    | Hyperbolic sine of input vector expression elements.        |
| \ref TNL::cosh    | Hyperbolic cosine of input vector expression elements.      |
| \ref TNL::tanh    | Hyperbolic tangent of input vector expression elements.     |
| \ref TNL::asinh   | Arc hyperbolic sine of input vector expression elements.    |
| \ref TNL::acosh   | Arc hyperbolic cosine of input vector expression elements.  |
| \ref TNL::atanh   | Arc hyperbolic tangent of input vector expression elements. |
| \ref TNL::exp     | Exponential function of input vector expression elements.   |
| \ref TNL::log     | Natural logarithm of input vector expression elements.      |
| \ref TNL::log10   | Decadic logarithm of input vector expression elements.      |
| \ref TNL::log2    | Binary logarithm of input vector expression elements.       |
| \ref TNL::sqrt    | Square root of input vector expression elements.            |
| \ref TNL::cbrt    | Cubic root of input vector expression elements.             |
| \ref TNL::pow     | Power of of input vector expression elements.               |
| \ref TNL::floor   | Rounds downward input vector expression elements.           |
| \ref TNL::ceil    | Rounds upward of input vector expression elements.          |
| \ref TNL::sign    | Signum of input vector expression elements.                 |

### Vertical operations

By *vertical operations* we mean (parallel) reduction based operations where we have one vector expressions as an input and one value as an output. For example computing scalar product, vector norm or finding minimum or maximum of vector elements is based on reduction. See the following example.

\includelineno Reduction.cpp

Output is:

\include Reduction.out

The following table shows vertical operations that can be used on vector expressions:

| Function             | Meaning                                                                   |
|----------------------|---------------------------------------------------------------------------|
| \ref TNL::min        | Minimum of vector expression elements.                                    |
| \ref TNL::argMin     | Minimum of vector expression elements with index of the smallest element. |
| \ref TNL::max        | Maximum of vector expression elements.                                    |
| \ref TNL::argMax     | Minimum of vector expression elements with index of the smallest element. |
| \ref TNL::sum        | Sum of vector expression elements.                                        |
| \ref TNL::maxNorm    | Maximal norm of vector expression elements.                               |
| \ref TNL::l1Norm     | l1 norm of vector expression elements.                                    |
| \ref TNL::l2Norm     | l2 norm of vector expression elements.                                    |
| \ref TNL::lpNorm     | lp norm of vector expression elements. `p` is given as second argument.   |
| \ref TNL::product    | Product of vector expression elements.                                    |
| \ref TNL::logicalAnd | Logical AND of vector expression elements.                                |
| \ref TNL::logicalOr  | Logical OR of vector expression elements.                                 |
| \ref TNL::binaryAnd  | Binary AND of vector expression elements.                                 |
| \ref TNL::binaryOr   | Binary OR of vector expression elements.                                  |

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
