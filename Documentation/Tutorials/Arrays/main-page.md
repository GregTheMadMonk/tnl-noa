# Arrays tutorial

## Introduction

This tutorial introduces arrays in TNL. Array is one of the most important structure for memory management. Methods implemented in arrays are particularly usefull for GPU programming. From this point of view, the reader will learn how to easily allocate memory on GPU, transfer data between GPU and CPU but also, how to initialise data allocated on GPU. In addition, the resulting code is hardware platform independent, so it can be ran on CPU without any changes.

## Arrays

Array is templated class defined in namespace ```TNL::Containers``` having three template parameters:

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

Because of the data sharing, TNL Array is relatively complicated structure. In many situations, we prefer lightweight structure which only encapsulates the data pointer and keeps information about the data size. Passing array structure to GPU kernel can be one example. For this purpose there is ```ArrayView``` in TNL. It templated structure having the same template parameters as ```Array``` (it means ```Value```, ```Device``` and ```Index```). In fact, it is recommended to use ```Array``` only for the data allocation and to use ```ArrayView``` for most of the operations with the data since array view offer better functionality (for example ```ArrayView``` can be captured by lambda functions in CUDA while ```Array``` cannot). The following code snippet shows how to create an array view.

\include ArrayView-1.cpp

Its output is:

\include ArrayView-1.out

Of course, one may bind his own data into array view:

\include ArrayView-2.cpp

Output:

\include ArrayView-2.out

Array view never allocated or deallocate the memory managed by it. Therefore it can be created even in CUDA kernels which is not true for ```Array```.

## Accessing the array elements

There are two ways how to work with the array (or array view) elements - using the indexing operator (```operator[]```) which is more efficient or methods ```setElement``` and ```getElement``` which is more flexible.

### Accessing the array elements with ```operator[]```

Indexing operator ```operator[]``` is implemented in both ```Array``` and ```ArrayView``` and it is defined as ```__cuda_callable__```. It means that it can be called even in CUDA kernels if the data is allocated on GPU, i.e. the ```Device``` parameter is ```Devicess::Cuda```. This operator returns a reference to given array element and so it is very efficient. However, calling this operator from host for data allocated in device (or vice versa) leads to segmentation fault (on the host system) or broken state of the device. It means:

* You may call the ```operator[]``` on the **host** only for data allocated on the **host** (with device ```Devices::Host```).
* You may call the ```operator[]``` on the **device** only for data allocated on the **device** (with device ```Devices::Cuda```).

The following example shows use of ```operator[]```.

\include ElementsAccessing-1.cpp

Output:

\include ElementsAccessing-1.out

In general in TNL, each method defined as ```__cuda_callable__``` can be called from the CUDA kernels. The method ```ArrayView::getSize``` is another example. We also would like to point the reader to better ways of arrays initiation for example with method ```ArrayView::evaluate``` or with ```ParalleFor```.

### Accessing the array element with ```setElement``` and ```getElement```

On the other hand, the methods ```setElement``` and ```getElement``` can be called **from the host only** no matter where the array is allocated. None of the methods can be used in CUDA kernels. ```getElement``` returns copy of an element rather than a reference. Therefore it is slightly slower. If the array is on GPU, the array element is copied from the device on the host (or vice versa) which is significantly slower. In those parts of code where the perfomance matters, these methods shall not be called. Their use is, however, much easier and they allow to write one simple code for both CPU and GPU. Both methods are good candidates for:

* reading/wiriting of only few elements in the array
* arrays inititation which is done only once and it is not time critical part of a code
* debugging purposes

The following example shows the use of ```getElement``` and ```setElement```:

\include ElementsAccessing-2.cpp

Output:

\include ElementsAccessing-2.out

## Arrays initiation with lambdas

More eifficient and still quite simple method for the arrays initiation is with the use of C++ lambda functions and method ```evaluate```. This method is implemented in ```ArrayView``` only. As an argument a lambda function is passed which is then evaluated for all elemeents. Optionaly one may define only subinterval of element indexes where the lambda shall be evaluated. If the underlaying array is allocated on GPU, the lambda function is called from CUDA kernel. This is why it is more efficient than use of ```setElement```. On the other hand, one must be carefull to use only ```__cuda_callable__``` methods inside the lambda. The use of the method ```evaluate``` demonstrates the following example.

\include ArrayViewEvaluate.cpp

Output:

\include ArrayViewEvaluate.out

## Checking the array contents

Methods ```containsValue``` and ```containsOnlyValue``` serve for testing the contents of the arrays. ```containsValue``` returns ```true``` of there is at least one element in the array with given value. ```containsOnlyValue``` returnd ```true``` only if all elements of the array equal given value. The test can be restricted to subinterval of array elements. Both methods are implemented in ```Array``` as well as in ```ArrayView```. See the following code snippet for example of use.

\include ContainsValue.cpp

Output:

\include ContainsValue.out

## IO operations with Arrays


