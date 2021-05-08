#include "../../../otherGPUsorts/cudaExamples/cdpAdvancedQuicksort/cdpAdvancedQuicksort.cu"
#include "../../../otherGPUsorts/cudaExamples/cdpAdvancedQuicksort/cdpBitonicSort.cu"
#include <TNL/Containers/Array.h>

//---------------------------
void sorter(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> view)
{
    TNL::Containers::Array<int, TNL::Devices::Cuda> aux(view.getSize());
    run_quicksort_cdp((unsigned int *)view.getData(), (unsigned int *)aux.getData(), view.getSize(), NULL);
}

#include "../../../GPUSort/benchmark/benchmarker.cpp"
#include "../../../GPUSort/benchmark/measure.cu"