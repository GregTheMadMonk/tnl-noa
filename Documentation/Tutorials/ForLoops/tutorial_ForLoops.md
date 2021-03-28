\page tutorial_ForLoops For loops

[TOC]

## Introduction

This tutorial shows how to use different kind of for loops implemented in TNL. Namely, they are:

* **Parallel for** is a for loop which can be run in parallel, i.e. all iterations of the loop must be independent. Paralle for can run on both multicore CPUs and GPUs.
* **n-dimensional Parallel For** is extension of common parallel for into more dimensions.
* **Static For** is a for loop which is performed sequentialy and it is explicitly unrolled by C++ templates. Number of iterations must be static (known at compile time).
* **Templated Static For** ....

## Parallel For

Basic parallel for construction in TNL serves for hardware platform transparent expression of parallel for loops. The hardware platform is expressed by a template parameter. The parallel for is defined as:

```
ParallelFor< Device >::exec( start, end, function, arguments... );
```

The `Device` can be either `Devices::Host` or `Devices::Cuda`. The first two parameters define the loop bounds in the C style. It means that there will be iterations for indexes `start` ... `end-1`. Function is a lambda function to be performed in each iteration. It is supposed to receive the iteration index and arguments passed to the parallel for (the last arguments). See the following example:

\include ParallelForExample_ug.cpp

The result is:

\include ParallelForExample.out

## n-dimensional Parallel For

Performing for-loops in higher dimensions is simillar. In the following example we build 2D mesh function on top of TNL vector. Two dimensional indexes `( i, j )` are mapped to vector index `idx` as `idx = j * xSize + i`, where the mesh fuction has dimensions `xSize * ySize`. Of course, in this simple example, it does not make any sense to compute a sum of two mesh function this way, it is only an example.

\include ParallelForExample-2D_ug.cpp

Notice the parameters of the lambda function `sum`. The first parameter `i` changes more often than `j` and therefore the index mapping has the form `j * xSize + i` to acces the vector elements sequentialy on CPU and to fullfill coalesced memory accesses on GPU. The for-loop is executed by calling `ParallelFor2D` with proper device. The first four parameters are `startX, startY, endX, endY` and on CPU this is equivalent to the following embeded for loops:

\include ParallelFor2D-snippet.cpp

where `args...` stand for additional arguments passed to the for-loop. After the parameters defining the loops bounds, lambda function (`sum` in this case) is passed followed by additional arguments. One of them, in our example, is `xSize` again because it must be passed to the lambda function for the index mapping computation.

For the completness, we show modification of the previous example into 3D:

\include ParallelForExample-3D_ug.cpp

## Static For

Static for-loop is designed for short loops with constant (i.e. known at the compile time) number of iterations. It is often used with static arrays and vectors. An adventage of this kind of for loop is that it is explicitly unrolled when the loop is short (up to eight iterations). See the following example:

\include StaticForExample_ug.cpp

Notice that the static for-loop works with a lambda function simillar to parallel for-loop. The bounds of the loop are passed as template parameters in the statement `Algorithms::StaticFor< 0, Size >`. The parameters of the static method `exec` are the lambda functions to be performed in each iteration and auxiliar data to be passed to the function. The function gets the loop index `i` first followed by the auxiliary data `sum` in this example.

The result looks as:

\include StaticForExample.out

The effect of `StaticFor` is really the same as usual for-loop. The following code does the same as the previous example:

\include StaticForExample-2.cpp

The benefit of `StaticFor` is mainly in the explicit unrolling of short loops which can improve the performance in some situations. `StaticFor` can be forced to do the loop-unrolling in any situations using the third template parameter as follows:

\include StaticForExample-3.cpp

`StaticFor` can be used also in CUDA kernels.

## Templated Static For

Templated static for-loop (`TemplateStaticFor`) is a for-loop in template parameters. For example, if class `LoopBody` is defined as

```
template< int i >
struct LoopBody
{
   static void exec() { ... };
}
```

one might need to execute the following sequence of statements:

```
LoopBody< 0 >::exec();
LoopBody< 1 >::exec();
LoopBody< 3 >::exec();
...
LoodBody< N >::exec();
```

This is exactly what `TemplateStaticFor` can do - in a slightly more general way. See the following example:

\include TemplateStaticForExample.cpp

The output looks as follows:

\include TemplateStaticForExample.out
