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
   8. [Reduction with argument](#flexible_parallel_reduction_with_argument)
2. [Flexible Scan](#flexible_scan)
   1. [Inclusive and exclusive scna](#inclusive_and_exclusive_scan)
   2. [Segmented scan](#segmented_scan)

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

Putting everything together gives the following example:

\include SumExample.cpp

Since TNL vectors cannot be pass to CUDA kernels and so they cannot be captured by CUDA lambdas, we must first get vector view from the vector using a method `getConstView()`.

Note tha we pass `0.0` as the last argument of the method `Reduction< Device >::reduce`. It is an *idempotent element* (see [Idempotence](https://cs.wikipedia.org/wiki/Idempotence)). It is an element which, for given operation, does not change the result. For addition, it is zero. The result looks as follows.

\include SumExample.out

Sum of vector elements can be also obtained as [`sum(v)`](../html/namespaceTNL.html#a41cea4796188f0877dbb6e72e2d3559e).

### Product<a name="flexible_parallel_reduction_product"></a>

To demonstrate the effect of the *idempotent element*, we will now compute product of all elements of the vector. The *idempotent element* is one for multiplication and we also need to replace `a+b` with `a*b` in the definition of `reduce`. We get the following code:

\include ProductExample.cpp

leading to output like this:

\include ProductExample.out

Product of vector elements can be computed using fuction [`product(v)`](../html/namespaceTNL.html#ac11e1901681d36b19a0ad3c6f167a718).

### Scalar product<a name="flexible_parallel_reduction_scalar_product"></a>

One of the most important operation in the linear algebra is the scalar product of two vectors. Compared to coputing the sum of vector elements we must change the function `fetch` to read elements from both vectors and multiply them. See the following example.

\include ScalarProductExample.cpp

The result is:

\include ScalarProductExample.out

Scalar product of vectors `u` and `v` in TNL can be computed by \ref TNL::dot "TNL::dot(u, v)" or simply as \ref TNL::Containers::operator, "(u, v)".

### Maxium norm<a name="flexible_parallel_reduction_maximum_norm"></a>

Maximum norm of a vector equals modulus of the vector largest element.  Therefore, `fetch` must return the absolute value of the vector elements and `reduce` wil return maximum of given values. Look at the following example.

\include MaximumNormExample.cpp

The output is:

\include MaximumNormExample.out

Maximum norm in TNL is computed by the function \ref TNL::maxNorm.

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

### Simple MapReduce<a name="flexible_parallel_reduction_simple_map_reduce"></a>

We can also filter the data to be reduced. This operation is called [MapReduce](https://en.wikipedia.org/wiki/MapReduce) . You simply add necessary if statement to the fetch function, or in the case of the following example we use a statement

```
return u_view[ i ] > 0.0 ? u_view[ i ] : 0.0;
```

to sum up only the positive numbers in the vector.

\include MapReduceExample-1.cpp

The result is:

\include MapReduceExample-1.out

Take a look at the following example where the filtering depends on the element indexes rather than values:

\include MapReduceExample-2.cpp

The result is:

\include MapReduceExample-2.out

This is not very efficient. For half of the elements, we return zero which has no effect during the reductin. Better solution is to run the reduction only for a half of the elements and to change the fetch function to

```
return u_view[ 2 * i ];
```

See the following example and compare the execution times.

\include MapReduceExample-3.cpp

\include MapReduceExample-3.out
 
### Reduction with argument<a name="flexible_parallel_reduction_with_argument"></a>

In some situations we may need to locate given element in the vector. For example index of the smallest or the largest element. `reductionWithArgument` is a function which can do it. In the following example, we modify function for computing the maximum norm of a vedctor. Instead of just computing the value, now we want to get index of the element having the absolute value equal to the max norm. The lambda function `reduction` do not compute only maximum of two given elements anymore, but it must also compute index of the winner. See the following code:

\include ReductionWithArgument.cpp

The definition of the lambda function `reduction` reads as:

```
auto reduction = [] __cuda_callable__ ( int& aIdx, const int& bIdx, double& a, const double& b );
```

In addition to vector elements valuesd `a` and `b`, it gets also their positions `aIdx` and `bIdx`. The functions is responsible to set `a` to maximum of the two and `aIdx` to the position of the larger element. Note, that the parameters have the above mentioned meaning only in case of computing minimum or maximum.

The result looks as:

\include ReductionWithArgument.out

## Flexible scan<a name="flexible_scan"></a>

### Inclusive and exclusive scan<a name="inclusive_and_exclusive_scan"></a>
Inclusive scan (or prefix sum) operation turns a sequence \f$a_1, \ldots, a_n\f$ into a sequence \f$s_1, \ldots, s_n\f$ defined as

\f[
s_i = \sum_{j=1}^i a_i.
\f]

Exclusive scan (or prefix sum) is defined as

\f[
\sigma_i = \sum_{j=1}^{i-1} a_i.
\f]

For example, inclusive prefix sum of

```
[1,3,5,7,9,11,13]
```

is

```
[1,4,9,16,25,36,49]
```

and exclusive prefix sum of the same sequence is

```
[0,1,4,9,16,25,36]
```

Both kinds of [scan](https://en.wikipedia.org/wiki/Prefix_sum) are usually applied only on sumation, however product or logical operations could be handy as well. In TNL, prefix sum is implemented in simillar way as reduction and so it can be easily modified by lambda functions. The following example shows how it works:

\include ScanExample.cpp

Scan does not use `fetch` function because the scan must be performed on a vector (the first parameter we pass to the scan). Its complexity is also higher compared to reduction. Thus if one needs to do some operation with the vector elements before the scan, this can be done explicitly and it will not affect the performance significantlty. On the other hand, the scan function takes interval of the vector elements where the scan is performed as its second and third argument. The next argument is the operation to be performed by the scan and the last parameter is the idempotent ("zero") element if the operation.

The result looks as:

\include ScanExample.out

Exclusive scan works the same way, we just need to specify it by the second template parameter which is set to `ScanType::Exclusive`. The call of the scan then looks as

```
Scan< Device, ScanType::Exclusive >::perform( v, 0, v.getSize(), reduce, 0.0 );
``` 

The complete example looks as follows:

\include ExclusiveScanExample.cpp

And the result looks as:

\include ExclusiveScanExample.out

### Segmented scan<a name="segmented_scan"></a>

Segmented scan is a modification of common scan. In this case the sequence of numbers in hand is divided into segments like this, for example

```
[1,3,5][2,4,6,9][3,5],[3,6,9,12,15]
```

and we want to compute inclusive or exclusive scan of each segment. For inclusive segmented prefix sum we get

```
[1,4,9][2,6,12,21][3,8][3,9,18,30,45]
```

and for exclusive segmented prefix sum it is

```
[0,1,4][0,2,6,12][0,3][0,3,9,18,30]
```

In addition to common scan, we need to encode the segments of the input sequence. It is done by auxiliary flags array (it can be array of booleans) having `1` at the begining of each segment and `0` on all other positions. In our example, it would be like this:

```
[1,0,0,1,0,0,0,1,0,1,0,0, 0, 0]
[1,3,5,2,4,6,9,3,5,3,6,9,12,15]
```
**Note: Segmented scan is not implemented for CUDA yet.**

\include SegmentedScanExample.cpp

The result reads as:

\include SegmentedScanExample.out
