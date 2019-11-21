\page tutorial_Pointers  Cross-device pointers tutorial

## Introduction

Smart pointers in TNL are motivated by the smart pointerin the STL library. In addition, they work across different devices and so they make data management easier.

## Table of Contents
1. [Unique pointers](#unique_pointers)
2. [Shared pointers](#shared_pointers)
3. [Device pointers](#device_pointers)


## Unique pointers <a name="unique_pointers"></a>

Simillar to STL smart pointer `std::unique_ptr` `UniquePointer` is a smart poinetr managing certain dynamicaly allocated object. The object is automatically  deallocated when the pointer goes out of scope. The definition of `UniquePointer` reads as:

```
template< typename Object, typename Device = typename Object::DeviceType >
class UniquePointer;
```

It takes two template parameters:

1. `Object` is a type of object managed by the pointer.
2. `Device` is a device where the object is to be allocated.

If the device type is `Devices::Host`, `UniquePointer` behaves as usual unique smart pointer. If the device is different, `Devices::Cuda` for example, the unique pointer creates an image if the object even in the host memory. It means, that one can manipulate the object on the host. All smart pointers are registered in a special register using which they can be easily synchronised before calling a CUDA kernel. This means that all modified images of the objects in the memory are transferred on the GPU. See the following example:

\include UniquePointerExample.cpp

The result looks as:

\include UniquePointerExample.out

## Shared pointers <a name="shared_pointers"></a>

## Device pointers <a name="device_pointers"></a>
