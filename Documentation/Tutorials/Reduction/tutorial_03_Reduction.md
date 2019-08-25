\page tutorial_03_reduction Flexible (parallel) reduction and prefix-sum tutorial

## Introduction

This tutorial introduces flexible parallel reduction in TNL. It shows how to easily implement parallel reduction with user defined operations which may run on both CPU and GPU. Parallel reduction is a programming pattern appering very often in different kind of algorithms for example in scalar product, vector norms or mean value evaluation but also in sequences or strings comparison.

## Table of Contents
1. [Flexible Parallel Reduction](#flexible_parallel_reduction)
   1. [Sum](#flexible_parallel_reduction_sum)
   2. [Product](#flexible_parallel_reduction_product)
   3. [Scalar product](#flexible_parallel_reduction_scalar_product)
   4. [Maxium norm](#flexible_parallel_reduction_maximum_norm)
   5. [Vectors comparison](#flexible_parallel_reduction_vector_comparison)
   6. [Update and Residue](#flexible_parallel_reduction_update_and_residue)
   7. [Simple Mask and Reduce](#flexible_parallel_reduction_simple_mask_and_reduce)
   8. [Using reduction in class methods](#flexible_parallel_reduction_using_reduction_in_class_methods)

## Flexible parallel reduction<a name="flexible_parallel_reduction"></a>

We will explain the *flexible parallel reduction* on several examples. We start with the simplest sum of sequence of numbers followed by more advanced problems like scalar product or vector norms.

### Sum<a name="flexible_parallel_reduction_sum"></a>

We start with simple problem of computing sum of sequence of numbers \f[ s = \sum_{i=1}^n a_i. \f] Sequentialy, such sum can be computed very easily as follows:

\include SequentialSum.cpp

Doing the same in CUDA for GPU is, however, much more difficult (see. [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)). The final code has tens of lines and it is something you do not want to write again and again anytime you need to sum a series of numbers. Using TNL and C++ lambda functions we may do the same on few lines of code efficiently and independently on the hardware beneath. Let us first rewrite the previous example using the C++ lambda functions:

\include SequentialSumWithLambdas.cpp

As can be seen, we split the reduction into two steps:

1. `fetch` reads the input data. Thanks to this lambda you can:
   1. Connect the reduction algorithm with given input arrays or vectors (or any other data structure).
   2. Perform operation you need to do with the input data.
   3. Perform another secondary operation simoultanously with the parallel reduction.
2. `reduce` is operation we want to do after the data fetch. Usually it is summation, multiplication, evaluation of minimum or maximum or some logical operation.

For the purpose of the CUDA parallel reduction, we need to provide also volatile variant of `reduce` (this could be hopefully avoided in future when `nvcc` supports generic lambdas better): 

\include SequentialSumWithLambdas-2.cpp

The difference is only in the lambda function parameters definition, they are `volatile` now.

Putting everything together gives the following example:

\include SumExample.cpp

Since TNL vectors cannot be pass to CUDA kernels and so they cannot be captured by CUDA lambdas, we must first get vector view from the vector using a method `getView()`.

Note tha we pass `0.0` as the last argument of the method `Reduction< Device >::reduce`. It is an *idempotent element* (see [Idempotence](https://cs.wikipedia.org/wiki/Idempotence)). It is an element which, for given operation, does not change the result. For addition, it is zero. The result looks as follows.

\include SumExample.out

Sum of vector elements can be also obtained as `sum(v)`.

### Product<a name="flexible_parallel_reduction_product"></a>

To demonstrate the effect of the *idempotent element*, we will now compute product of all elements of the vector. The *idempotent element* is one for multiplication and we also need to replace `a+=b` with `a*=b` in the definition of `reduce` and `volatileReduce`. We get the following code:
i
\include ProductExample.cpp

leading to output like this:

\include ProductExample.out

Product of vector elements can be computed using fuction `product(v)`.

### Scalar product<a name="flexible_parallel_reduction_scalar_product"></a>

One of the most important operation in the linear algebra is the scalar product of two vectors. Compared to coputing the sum of vector elements we must change the function `fetch` to read elements from both vectors and multiply them. See the following example.

\include ScalarProductExample.cpp

The result is:

\include ScalarProductExample.out

Scalar product of vectors `u` and `v` can be in TNL computed by `dot(u,v)` or simply as `(u,v)`.

### Maxium norm<a name="flexible_parallel_reduction_maximum_norm"></a>

Maximum norm of a vector equals modulus of the vector largest element.  Therefore, `fetch` must return the absolute value of the vector elements and `reduce` wil return maximum of given values. Look at the following example.

\include MaximumNormExample.cpp

The output is:

\include MaximumNormExample.out

Maximum norm in TNL computes function `maxNorm(v)`.

### Vectors comparison<a name="flexible_parallel_reduction_vector_comparison"></a>

Comparison of two vectors involve (parallel) reduction as well. The `fetch` part is responsible for comparison of corresponding vector elements result of which is boolean `true` or `false` for each vector elements. The `reduce` part must perform logical and operation on all of them. We must not forget to change the *idempotent element* to `true`. The code may look as follows:

\include ComparisonExample.cpp

And the output looks as:

\include ComparisonExample.out

### Update and residue<a name="flexible_parallel_reduction_update_and_residue"></a>

In iterative solvers we often need to update a vector and compute the update norm at the same time. For example the [Euler method](https://en.wikipedia.org/wiki/Euler_method) is defined as 

\f[
\bf u^{k+1} = \bf u^k + \tau \Delta \bf u.
\f]

Together with the vector addition, we may want to compute also \f$L_2\f$-norm of \f$\Delta \bf u\f$ which may indicate convergence. Computing first the addition and then the norm would be inefficient because we would have to fetch the vector \f$\Delta \bf u\f$ twice from the memory. The following example shows how to do the addition and norm computation at the same time.

\include UpdateAndResidueExample.cpp

The result reads as:

\include UpdateAndResidueExample.out

### Simple Mask and Reduce<a name="flexible_parallel_reduction_simple_mask_and_reduce"></a>

We can also filter the data to be reduced. This operation is called Mask and Reduce TODO: REF. You simply add necessary if statement to the fetch function, or in the case of the following example we use a statement

```
return u_view[ i ] > 0.0 ? u_view[ i ] : 0.0;
```

to sum up only the positive numbers in the vector.

\include MaskAndReduceExample-1.cpp

The result is:

\include MaskAndReduceExample-1.out

Take a look at the following example where the filtering depends on the element indexes rather than values:

\include MaskAndReduceExample-2.cpp

The result is:

\include MaskAndReduceExample-2.out

This is not very efficient. For half of the elements, we return zero which has no effect during the reductin. Better solution is to run the reduction only for a half of the elements and to change the fetch function to

```
return u_view[ 2 * i ];
```

See the following example and compare the execution times.

\include MaskAndReduceExample-3.cpp

\include MaskAndReduceExample-3.out
 
### Using reduction in class methods<a name="flexible_parallel_reduction_using_reduction_in_class_methods"></a>

TODO:
