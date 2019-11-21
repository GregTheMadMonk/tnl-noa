\page tutorial_Pointers  Cross-device pointers tutorial

## Introduction

Smart pointers in TNL are motivated by the smart pointerin the STL library. In addition, they can manage image of the object they hold on different devices which makes objects offloading easier.

## Table of Contents
1. [Unique pointers](#unique_pointers)
2. [Shared pointers](#shared_pointers)
3. [Device pointers](#device_pointers)


## Unique pointers <a name="unique_pointers"></a>

Simillar to STL smart pointer `std::unique_ptr` `UniquePointer` is a smart pointer managing certain dynamicaly allocated object. The object is automatically deallocated when the pointer goes out of scope. The definition of `UniquePointer` reads as:

```
template< typename Object, typename Device = typename Object::DeviceType >
class UniquePointer;
```

It takes two template parameters:

1. `Object` is a type of object managed by the pointer.
2. `Device` is a device where the object is to be allocated.

If the device type is `Devices::Host`, `UniquePointer` behaves as usual unique smart pointer. See the following example:

\include UniquePointerHostExample.cpp

The result is:

\include UniquePointerHostExample.out


If the device is different, `Devices::Cuda` for example, the unique pointer creates an image if the object even in the host memory. It means, that one can manipulate the object on the host. All smart pointers are registered in a special register using which they can be easily synchronised with the host images before calling a CUDA kernel. This means that all modified images of the objects in the memory are transferred on the GPU. See the following example:

\include UniquePointerExample.cpp

The result looks as:

\include UniquePointerExample.out

A disadventage of `UniquePointer` is that it cannot be passed to the CUDA kernel since it requires making a copy of it. This is, however, from the nature of this object, prohibited. Not only this is solved by a `SharedPointer`.

## Shared pointers <a name="shared_pointers"></a>

One of the main goals of the TNL library is to make the development of the HPC code, including GPU kernels as easy and efficient as possible. One way to do this is to profit from the object opriented programming even in CUDA kernels. Let us explain it on arrays. From certain point of view `Array` can be understood as an object consisiting of data and metadata. Data part means elements that we insert into the array. Metadata is a pointer to the data but also size of the array. This information makes use of the class easier. Though it is not necessary in any situations it may help to check array bounds when accessing the array elements for example. It is something that, when it is performed even in CUDA kernels, may help significantly with finding bugs in a code.  To do this, we need to transfer on the GPU not only pointers to the data but also complete metadata. It is simple if the structure which is supposed to be transfered on the GPU does not have pointers to metadata. See the following example:

```
struct Array
{
   double* data;
   int size;
};
```

If the pointer `data` points to a memory on GPU, this array can be passed to a kernel like this:

```
Array a;
cudaKernel<<< gridSize, blockSize >>>( a );
```

The kernel `cudaKernel` can access the data as follows:

```
__global__ void cudaKernel( Array a )
{
   if( thredadIdx.x. < a.size )
      a.data[ threadIdx.x ] = 0;
}
```

But what if we have an object like this:

```
struct ArrayTuple
{
   Array *a1, *a2;
}
```

Assume that there is an instance of `ArrayTuple` lets say `tuple` containing pointers to instances `a1` and `a2` of `Array`. The instances must be allocated on the GPU if one wants to simply pass the `tuple` to the CUDA kernel. Indeed, the CUDA kernels needs the arrays `a1` and `a2` to be on the GPU. See the following example:

```
__global__ tupleKernel( ArrayTuple tuple )
{
   if( threadIdx.x < tuple.a1->size )
      tuple.a1->data[ threadIdx.x ] = 0;
   if( threadIdx.x < tuple.a2->size )
      tuple.a2->data[ threadIdx.x ] = 0;
}

```

See, that the kernel needs to dereference `tuple.a1` and `tuple.a2`. Therefore these pointers must point to the global memoty of the GPU which means that arrays `a1` and `a2` must be allocated there using [cudaMalloc](http://developer.download.nvidia.com/compute/cuda/2_3/toolkit/docs/online/group__CUDART__MEMORY_gc63ffd93e344b939d6399199d8b12fef.html) lets say. It means, however, that the arrays `a1` and `a2` cannot be managed (for example resizing them requires changing `a1->size` and `a2->size`) on the host system by the CPU. The only solution to this is to have images of `a1` and `a2` and in the host memory and to copy them on the GPU before calling the CUDA kernel. One must not forget to modify the pointers in the `tuple` to point to the array copies on the GPU. To simplify this, TNL offers *cross-device shared smart pointers*. In addition to common smart pointers thay can manage an images of an object on different devices. Note that [CUDA Unified Memory](https://devblogs.nvidia.com/unified-memory-cuda-beginners/) is an answer to this problem as well. TNL cross-device smart pointers can be more efficient in some situations. (TODO: Prove this with benchmark problem.)

The previous example could be implemented in TNL as follows:

\include SharedPointerExample.cpp

The result looks as:

\include SharedPointerExample.out

## Device pointers <a name="device_pointers"></a>
