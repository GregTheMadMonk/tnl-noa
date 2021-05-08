## repository for bachelor thesis on Development of parallel sorting algorithms for GPU


# directory structure
* measuring
    * scripts and codes used to make comparison between different algorithms
* otherGPUsorts
    * code of other sorting algorithms
* GPUSort
    * implementation Bitonic sort and Quick sort for the thesis


sidenote:

warnings during compilation such as the one below are emitted by the TNL library and is an expected behaviour

/home/<user>/.local/include/TNL/Containers/ArrayView.h(155): warning: __host__ annotation is ignored on a function("ArrayView") that is explicitly defaulted on its first declaration