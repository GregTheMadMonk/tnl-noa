#include <TNL/Containers/Array.h>
#include "../../otherGPUsorts/manca_quicksort_extracted/manca_quicksort.cu"
//------------------------

void sorter(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> view)
{
    double timer = 0;
    CUDA_Quicksort((unsigned *)view.getData(), (unsigned *)view.getData(), view.getSize(), 256, 0, &timer);
    return;
}

//------------------------

#include "../../GPUSort/benchmark/benchmarker.cpp"
#include "../../GPUSort/benchmark/measure.cu"
