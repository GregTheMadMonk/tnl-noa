#include <TNL/Containers/Array.h>
#include "../../../otherGPUsorts/davors/BitonicSort/Sort/parallel.h"
//------------------------

#define LOW_POW 19
#define HIGH_POW 20

void sorter(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> view)
{
    auto sorter = new BitonicSortParallel();
    sorter->sort((data_t*)view.getData(), (uint_t)view.getSize(), ORDER_ASC);
    cudaDeviceSynchronize();
    delete sorter;
    return;
}

//------------------------

#include "../../../GPUSort/benchmark/benchmarker.cpp"
#include "../../../GPUSort/benchmark/measure.cu"
