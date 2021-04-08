\page tutorial_ForLoops For loops

[TOC]

## Introduction

This tutorial shows how to use different kind of for-loops implemented in TNL. Namely, they are:

* **Parallel for** is a for-loop which can be run in parallel, i.e. all iterations of the loop must be independent. Parallel for can be run on both multicore CPUs and GPUs.
* **n-dimensional parallel for** is an extension of common parallel for into higher dimensions.
* **Static For** is a for loop which is performed sequentialy and it is explicitly unrolled by C++ templates. Number of iterations must be static (known at compile time).
* **Templated Static For** ....

## Parallel For

Basic _parallel for_ construction in TNL serves for hardware platform transparent expression of parallel for-loops.
The hardware platform is specified by a template parameter.
The loop is implemented as \ref TNL::Algorithms::ParallelFor and can be used as:

```
ParallelFor< Device >::exec( start, end, function, arguments... );
```

The `Device` can be either \ref TNL::Devices::Host or \ref TNL::Devices::Cuda.
The first two parameters define the loop bounds in the C style.
It means that there will be iterations for indices `start`, `start+1`, ..., `end-1`.
The `function` is a lambda function to be called in each iteration.
It is supposed to receive the iteration index and arguments passed to the _parallel for_ (the last arguments).

See the following example:

\include ParallelForExample.cpp

The result is:

\include ParallelForExample.out

## n-dimensional Parallel For

For-loops in higher dimensions can be performed similarly via \ref TNL::Algorithms::ParallelFor2D and \ref TNL::Algorithms::ParallelFor3D.
In the following example we build a 2D mesh function on top of \ref TNL::Containers::Vector.
Two dimensional indices `( i, j )` are mapped to the vector index `idx` as `idx = j * xSize + i`, where the mesh function has dimensions `xSize * ySize`.
The following simple example performs initiation of the mesh function with a constant value `c = 1.0`:

\include ParallelForExample-2D.cpp

Notice the parameters of the lambda function `init`.
The first parameter `i` changes more often than `j` and therefore the index mapping has the form `j * xSize + i` to access the vector elements sequentially on CPU and to fulfill coalesced memory accesses on GPU.
The for-loop is executed by calling `ParallelFor2D` with proper device.
The first four parameters are `startX, startY, endX, endY` and on CPU this is equivalent to the following embedded for-loops:

```cpp
for( Index j = startY; j < endY; j++ )
   for( Index i = startX; i < endX; i++ )
      f( i, j, args... );
```

where `args...` stand for additional arguments passed to the for-loop.
After the parameters defining the loops bounds, lambda function (`init` in this case) is passed, followed by additional arguments that are forwarded to the lambda function after the iteration indices.
In the example above there are no additional arguments, since the lambda function `init` captures all variables it needs to work with.

For completeness, we show modification of the previous example into 3D:

\include ParallelForExample-3D.cpp

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
