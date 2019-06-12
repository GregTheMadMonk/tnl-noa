# Arrays tutorial

## Introduction

This tutorial introduces arrays in TNL. Array is one of the most important structure for memory management. Methods implemented in arrays are particularly useful for GPU programming. From this point of view, the reader will learn how to easily allocate memory on GPU, transfer data between GPU and CPU but also, how to initialize data allocated on GPU. In addition, the resulting code is hardware platform independent, so it can be ran on CPU without any changes.

## Table of Contents
1. [Arrays](#arrays)
   1. [Arrays binding](#arrays_binding)
   2. [Array views](#array_views)
   3. [Accessing the array elements](#accessing_the_array_elements)
      1. [Accessing the array elements with `operator[]`](#accessing_the_array_elements_with_operator)
      2. [Accessing the array elements with `setElement` and `getElement`](#accessing_the_array_elements_with_set_get_element)
   4. [Arrays initiation with lambdas](#arrays_initiation_with_lambdas)
   5. [Checking the array contents](#checking_the_array_contents)
   6. [IO operations with arrays](#io_operations_with-arrays)
2. [Static arrays](#static_arrays)

## Arrays <a name="arrays"></a>

Array is templated class defined in namespace `TNL::Containers` having three template parameters:

* `Value` is type of data to be stored in the array
* `Device` is the device where the array is allocated. Currently it can be either `Devices::Host` for CPU or `Devices::Cuda` for GPU supporting CUDA.
* `Index` is the type to be used for indexing the array elements.

The following example shows how to allocate arrays on CPU and GPU and how to initialize the data.

\include ArrayAllocation.cpp

The result looks as follows:

\include ArrayAllocation.out


### Arrays binding <a name="arrays_binding"></a>

Arrays can share data with each other or data allocated elsewhere. It is called binding and it can be done using method `bind`. The following example shows how to bind data allocated on host using the `new` operator. In this case, the TNL array do not free this data at the and of its life cycle.

\include ArrayBinding-1.cpp

It generates output like this:

\include ArrayBinding-1.out

One may also bind TNL array with another TNL array. In this case, the data is shared and can be shared between multiple arrays. Reference counter ensures that the data is freed after the last array sharing the data ends its life cycle. 

\include ArrayBinding-2.cpp

The result is:

\include ArrayBinding-2.out

Binding may also serve for data partitioning. Both CPU and GPU prefer data allocated in large contiguous blocks instead of many fragmented pieces of allocated memory. Another reason why one might want to partition the allocated data is demonstrated in the following example. Consider a situation of solving incompressible flow in 2D. The degrees of freedom consist of density and two components of velocity. Mostly, we want to manipulate either density or velocity. But some numerical solvers may need to have all degrees of freedom in one array. It can be managed like this:

\include ArrayBinding-3.cpp

The result is:

\include ArrayBinding-3.out


### Array views <a name="array_views"></a>

Because of the data sharing, TNL Array is relatively complicated structure. In many situations, we prefer lightweight structure which only encapsulates the data pointer and keeps information about the data size. Passing array structure to GPU kernel can be one example. For this purpose there is `ArrayView` in TNL. It is templated structure having the same template parameters as `Array` (it means `Value`, `Device` and `Index`). In fact, it is recommended to use `Array` only for the data allocation and to use `ArrayView` for most of the operations with the data since array view offer better functionality (for example `ArrayView` can be captured by lambda functions in CUDA while `Array` cannot). The following code snippet shows how to create an array view.

\include ArrayView-1.cpp

The output is:

\include ArrayView-1.out

Of course, one may bind his own data into array view:

\include ArrayView-2.cpp

Output:

\include ArrayView-2.out

Array view never allocated or deallocate the memory managed by it. Therefore it can be created even in CUDA kernels which is not true for `Array`.

### Accessing the array elements <a name="accessing_the_array_elements"></a>

There are two ways how to work with the array (or array view) elements - using the indexing operator (`operator[]`) which is more efficient or using methods `setElement` and `getElement` which is more flexible.

#### Accessing the array elements with `operator[]` <a name="accessing_the_array_elements_with_operator"></a>

Indexing operator `operator[]` is implemented in both `Array` and `ArrayView` and it is defined as `__cuda_callable__`. It means that it can be called even in CUDA kernels if the data is allocated on GPU, i.e. the `Device` parameter is `Devices::Cuda`. This operator returns a reference to given array element and so it is very efficient. However, calling this operator from host for data allocated on device (or vice versa) leads to segmentation fault (on the host system) or broken state of the device. It means:

* You may call the `operator[]` on the **host** only for data allocated on the **host** (with device `Devices::Host`).
* You may call the `operator[]` on the **device** only for data allocated on the **device** (with device `Devices::Cuda`).

The following example shows use of `operator[]`.

\include ElementsAccessing-1.cpp

Output:

\include ElementsAccessing-1.out

In general in TNL, each method defined as `__cuda_callable__` can be called from the CUDA kernels. The method `ArrayView::getSize` is another example. We also would like to point the reader to better ways of arrays initiation for example with method `ArrayView::evaluate` or with `ParallelFor`.

#### Accessing the array element with `setElement` and `getElement` <a name="accessing_the_array_elements_with_set_get_element"></a>

On the other hand, the methods `setElement` and `getElement` can be called **from the host only** no matter where the array is allocated. None of the methods can be used in CUDA kernels. `getElement` returns copy of an element rather than a reference. Therefore it is slightly slower. If the array is on GPU, the array element is copied from the device on the host (or vice versa) which is significantly slower. In the parts of code where the performance matters, these methods shall not be called. Their use is, however, much easier and they allow to write one simple code for both CPU and GPU. Both methods are good candidates for:

* reading/writing of only few elements in the array
* arrays initiation which is done only once and it is not time critical part of a code
* debugging purposes

The following example shows the use of `getElement` and `setElement`:

\include ElementsAccessing-2.cpp

Output:

\include ElementsAccessing-2.out

### Arrays initiation with lambdas <a name="arrays_inititation_with_lambdas"></a>

More efficient and still quite simple method for the arrays initiation is with the use of C++ lambda functions and method `evaluate`. This method is implemented in `ArrayView` only. As an argument a lambda function is passed which is then evaluated for all elements. Optionally one may define only subinterval of element indexes where the lambda shall be evaluated. If the underlying array is allocated on GPU, the lambda function is called from CUDA kernel. This is why it is more efficient than use of `setElement`. On the other hand, one must be careful to use only `__cuda_callable__` methods inside the lambda. The use of the method `evaluate` demonstrates the following example.

\include ArrayViewEvaluate.cpp

Output:

\include ArrayViewEvaluate.out

### Checking the array contents <a name="arrays"></a>

Methods `containsValue` and `containsOnlyValue` serve for testing the contents of the arrays. `containsValue` returns `true` of there is at least one element in the array with given value. `containsOnlyValue` returns `true` only if all elements of the array equal given value. The test can be restricted to subinterval of array elements. Both methods are implemented in `Array` as well as in `ArrayView`. See the following code snippet for example of use.

\include ContainsValue.cpp

Output:

\include ContainsValue.out

### IO operations with arrays <a name="arrays"></a>

Methods `save` and `load` serve for storing/restoring the array to/from a file in a binary form. In case of `Array`, loading of an array from a file causes data reallocation. `ArrayView` cannot do reallocation, therefore the data loaded from a file is copied to the memory managed by the `ArrayView`. The number of elements managed by the array view and those loaded from the file must be equal. See the following example.

\include ArrayIO.cpp

Output:

\include ArrayIO.out

## Static arrays <a name="static_arrays"></a>
