\page tutorial_Arrays  Arrays tutorial

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Arrays<a name="arrays"></a>](#arrays)
  - [Array views<a name="array-views"></a>](#array-views)
  - [Accessing the array elements<a name="accessing-the-array-elements"></a>](#accessing-the-array-elements)
    - [Accessing the array elements with `operator[]`<a name="accessing-the-array-elements-with-operator"></a>](#accessing-the-array-elements-with-operator)
    - [Accessing the array elements with `setElement` and `getElement`<a name="accessing-the-array-elements-with-setelement-and-getelement"></a>](#accessing-the-array-elements-with-setelement-and-getelement)
  - [Arrays initiation with lambdas<a name="arrays-initiation-with-lambdas"></a>](#arrays-initiation-with-lambdas)
  - [Checking the array contents<a name="checking-the-array-contents"></a>](#checking-the-array-contents)
  - [IO operations with arrays<a name="io-operations-with-arrays"></a>](#io-operations-with-arrays)
- [Static arrays<a name="static-arrays"></a>](#static-arrays)
- [Distributed arrays<a name="distributed-arrays"></a>](#distributed-arrays)

## Introduction

This tutorial introduces arrays in TNL. There are three types - common arrays with dynamic allocation, static arrays allocated on stack and distributed arrays with dynamic allocation. Arrays are one of the most important structures for memory management. Methods implemented in arrays are particularly useful for GPU programming. From this point of view, the reader will learn how to easily allocate memory on GPU, transfer data between GPU and CPU but also, how to initialize data allocated on GPU. In addition, the resulting code is hardware platform independent, so it can be ran on CPU nad GPU without any changes.

## Arrays<a name="arrays"></a>

Array is templated class defined in namespace `TNL::Containers` having three template parameters:

* `Value` is type of data to be stored in the array
* `Device` is the device where the array is allocated. Currently it can be either `Devices::Host` for CPU or `Devices::Cuda` for GPU supporting CUDA.
* `Index` is the type to be used for indexing the array elements.

The following example shows how to allocate arrays on CPU and GPU and how to initialize the data.

\include ArrayAllocation.cpp

The result looks as follows:

\include ArrayAllocation.out


### Array views<a name="array-views"></a>

Arrays cannot share data with each other or data allocated elsewhere. This can be achieved with the `ArrayView` structure which has similar semantics to `Array`, but it does not handle allocation and deallocation of the data. Hence, array view cannot be resized, but it can be used to wrap data allocated elsewhere (e.g. using an `Array` or an operator `new`) and to partition large arrays into subarrays. The process of wrapping external data with a view is called _binding_.

The following code snippet shows how to create an array view.

\include ArrayView-1.cpp

The output is:

\include ArrayView-1.out

You can also bind external data into array view:

\include ArrayView-2.cpp

Output:

\include ArrayView-2.out

Since array views do not allocate or deallocate memory, they can be created even in CUDA kernels, which is not possible with `Array`. `ArrayView` can also be passed-by-value into CUDA kernels or captured-by-value by device lambda functions, because the `ArrayView`'s copy-constructor makes only a shallow copy (i.e., it copies only the data pointer and size).

### Accessing the array elements<a name="accessing-the-array-elements"></a>

There are two ways how to work with the array (or array view) elements - using the indexing operator (`operator[]`) which is more efficient or using methods `setElement` and `getElement` which is more flexible.

#### Accessing the array elements with `operator[]`<a name="accessing-the-array-elements-with-operator"></a>

Indexing operator `operator[]` is implemented in both `Array` and `ArrayView` and it is defined as `__cuda_callable__`. It means that it can be called even in CUDA kernels if the data is allocated on GPU, i.e. the `Device` parameter is `Devices::Cuda`. This operator returns a reference to given array element and so it is very efficient. However, calling this operator from host for data allocated on device (or vice versa) leads to segmentation fault (on the host system) or broken state of the device. It means:

* You may call the `operator[]` on the **host** only for data allocated on the **host** (with device `Devices::Host`).
* You may call the `operator[]` on the **device** only for data allocated on the **device** (with device `Devices::Cuda`).

The following example shows use of `operator[]`.

\include ElementsAccessing-1.cpp

Output:

\include ElementsAccessing-1.out

In general in TNL, each method defined as `__cuda_callable__` can be called from the CUDA kernels. The method `ArrayView::getSize` is another example. We also would like to point the reader to better ways of arrays initiation for example with method `ArrayView::evaluate` or with `ParallelFor`.

#### Accessing the array elements with `setElement` and `getElement`<a name="accessing-the-array-elements-with-setelement-and-getelement"></a>

On the other hand, the methods `setElement` and `getElement` can be called from the host **no matter where the array is allocated**. In addition they can be called from kernels on device where the array is allocated. `getElement` returns copy of an element rather than a reference. Therefore it is slightly slower. If the array is on GPU and the methods are called from the host, the array element is copied from the device on the host (or vice versa) which is significantly slower. In the parts of code where the performance matters, these methods shall not be called from the host when the array is allocated on the device. In this way, their use is, however, easier compared to `operator[]` and they allow to write one simple code for both CPU and GPU. Both methods are good candidates for:

* reading/writing of only few elements in the array
* arrays initiation which is done only once and it is not time critical part of a code
* debugging purposes

The following example shows the use of `getElement` and `setElement`:

\include ElementsAccessing-2.cpp

Output:

\include ElementsAccessing-2.out

### Arrays initiation with lambdas<a name="arrays-initiation-with-lambdas"></a>

More efficient and still quite simple method for the arrays initiation is with the use of C++ lambda functions and method `evaluate`. This method is implemented in `ArrayView` only. As an argument a lambda function is passed which is then evaluated for all elements. Optionally one may define only subinterval of element indexes where the lambda shall be evaluated. If the underlying array is allocated on GPU, the lambda function is called from CUDA kernel. This is why it is more efficient than use of `setElement`. On the other hand, one must be careful to use only `__cuda_callable__` methods inside the lambda. The use of the method `evaluate` demonstrates the following example.

\include ArrayViewEvaluate.cpp

Output:

\include ArrayViewEvaluate.out

### Checking the array contents<a name="checking-the-array-contents"></a>

Methods `containsValue` and `containsOnlyValue` serve for testing the contents of the arrays. `containsValue` returns `true` of there is at least one element in the array with given value. `containsOnlyValue` returns `true` only if all elements of the array equal given value. The test can be restricted to subinterval of array elements. Both methods are implemented in `Array` as well as in `ArrayView`. See the following code snippet for example of use.

\include ContainsValue.cpp

Output:

\include ContainsValue.out

### IO operations with arrays<a name="io-operations-with-arrays"></a>

Methods `save` and `load` serve for storing/restoring the array to/from a file in a binary form. In case of `Array`, loading of an array from a file causes data reallocation. `ArrayView` cannot do reallocation, therefore the data loaded from a file is copied to the memory managed by the `ArrayView`. The number of elements managed by the array view and those loaded from the file must be equal. See the following example.

\include ArrayIO.cpp

Output:

\include ArrayIO.out

## Static arrays<a name="static-arrays"></a>

Static arrays are allocated on stack and thus they can be created even in CUDA kernels. Their size is fixed and it is given by a template parameter. Static array is a templated class defined in namespace `TNL::Containers` having two template parameters:

* `Size` is the array size.
* `Value` is type of data stored in the array.

The interface of StaticArray is very smillar to Array but much simpler. It contains set of common constructors. Array elements can be accessed by the `operator[]` and also using method `x()`, `y()` and `z()` when it makes sense. See the following example for typical use of StaticArray.

\include StaticArrayExample.cpp

The output looks as:

\include StaticArrayExample.out

## Distributed arrays<a name="distributed-arrays"></a>
