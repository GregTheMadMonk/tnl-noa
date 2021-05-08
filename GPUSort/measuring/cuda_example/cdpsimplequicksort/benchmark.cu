#include "../../../otherGPUsorts/cudaExamples/cdpSimpleQuicksort/cdpSimpleQuicksort.cu"
#include <TNL/Containers/Array.h>

#define SORTERFUNCTION nvidia_quick

#define HIGH_POW 20
//---------------------------

void nvidia_quick(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> view)
{
    run_qsort((unsigned int *)view.getData(), view.getSize());
    cudaDeviceSynchronize();
}

#include "../../../GPUSort/benchmark/benchmarker.cpp"