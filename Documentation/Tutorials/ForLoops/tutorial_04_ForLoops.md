\page tutorial_04_ForLoops For loops

## Introduction

This tutorial shows how to use different kind of for loops implemented in TNL. Namely, they are:

* **Parallel for** is a for loop which can be run in parallel, i.e. all iterations of the loop must be independent. Paralle for can run on both multicore CPUs and GPUs.
* **n-dimensional Parallel For** is extension of common parallel for into more dimensions.
* **Static For** is a for loop which is performed sequentialy and it is explicitly unrolled by C++ templates. Number of iterations must be static (known at compile time).
* **Templated Static For** ....

## Table of Contents
1. [Parallel For](#parallel_for)
2. [n-dimensional Parallel For](#n_dimensional_parallel_for)
3. [Static For](#static_for)
4. [Templated Static For](#templated_static_for)

## Parallel For<a name="parallel_for"></a>

Basic parallel for construction in TNL serves for hardware platform transparent expression of parallel for loops. The hardware platform is expressed by a template parameter. The parallel for is defined as:

```
ParallelFor< Device >::exec( start, end, function, arguments... );
```

The `Device` can be either `Devices::Host` or `Devices::Cuda`. The first two parameters define the loop bounds in the C style. It means that there will be iterations for indexes `start` ... `end-1`. Function is a lambda function to be performed in each iteration. It is supposed to receive the iteration index and arguments passed to the parallel for (the last arguments). See the following example:

\include ParallelForExample.cpp

The result is:

\include ParallelForExample.out 

## n-dimensional Parallel For<a name="n_dimensional_parallel_for"></a>
## Static For<a name="static_for"></a>
## Templated Static For<a name="templated_static_for"></a>

