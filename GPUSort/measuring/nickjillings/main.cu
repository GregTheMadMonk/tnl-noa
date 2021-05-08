
#include "../../otherGPUsorts/nickjillings/BitonicSortCUDA.cu"
#include <TNL/Containers/Array.h>


void sorter(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> view)
{
    BitonicSort::BitonicSortCUDA((unsigned int *)view.getData(), view.getSize());
}
//---------------------------
#include "../../GPUSort/benchmark/benchmarker.cpp"
#include "../../GPUSort/benchmark/measure.cu"