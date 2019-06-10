# Arrays tutorial

## Introduction

This tutorial introduces arrays and vectors in TNL. Array is one of the most important structure for memory management. Vector, in addition, offers also basic operations from linear algebra. Methods implemented in arrays and vectors are particularly usefull for GPU programming. From this point of view, the reader will learn how to easily allocate memory on GPU, transfer data between GPU and CPU but also, how to initialise data allocated on GPU and perform parallel reduction and vector operations without writting low-level CUDA kernels. In addition, the resulting code is hardware platform independent, so it can be ran on CPU without any changes.

## Arrays

Array is templated class define in namespace ```TNL::Containers``` having three template parameters:

* ```Value``` is type of data to be stored in the array
* ```Device``` is the device wheer the array is allocated. Currently it can be either ```Devices::Host``` for CPU or ```Devices::Cuda``` for GPU supporting CUDA.
* ```Index``` is the type to be used for indexing the array elements.

The following example shows how to allocate arrays on CPU and GPU and how to manipulate the data.

\include ArrayAllocation.cpp

The result looks as follows:

\include ArrayAllocation.out


## Arrays binding

Arrays can share data with each other or data allocated elsewhere. It is called binding and it can be done using method ```bind```. The following example shows how to bind data allocated on host using the ```new``` operator. In this case, the TNL array do not free this data at the and of its life cycle.

\include ArrayBinding-1.cpp

It generates output like this:

\include ArrayBinding-1.out

One may also bind another TNL array. In this case, the data is shared and can be shared between multiple arrays. Reference counter ensures that the data is freed after the last array sharing the data ends its life cycle. 

\include ArrayBinding-2.cpp

The result is:

\include ArrayBinding-2.out

Binding may also serve for data partitioning. Both CPU and GPU prefere data allocated in large contiguous blocks instead of many fragmented pieces of allocated memory. Another reason why one might want to partition the allocated data is demonstrated in the following example. Consider a situation of solving incompressible flow in 2D. The degrees of freedom consist of density and two components of velocity. Mostly, we want to manipulate either density or velocity. But some numerical solvers may need to have all degrees of freedom in one array. It can be managed like this:

\include ArrayBinding-3.cpp

The result is:

\include ArrayBinding-3.out


## Array views

Because of the data sharing, TNL Array is relatively complicated structure. In many situations, we prefer lightweight structure which only encapsulates the data pointer and keeps information about the data size. Passing array structure to GPU kernel can be one example. For this purpose there is ```ArrayView``` in TNL. It templated structure having the same template parameters as ```Array``` (it means ```Value```, ```Device``` and ```Index```). In fact, it is recommended to use ```Array``` only for the data allocation and to use ```ArrayView``` for most of the operations with the data since array view offer better functionality (for example ```ArrayView``` can be captured by lambda functions in CUDA while ```Array``` cannot).
