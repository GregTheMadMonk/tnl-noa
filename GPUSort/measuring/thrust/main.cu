#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <TNL/Containers/Array.h>

void sorter(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> view)
{
    thrust::sort(thrust::device, view.getData(), view.getData() + view.getSize());
    cudaDeviceSynchronize();
}
//---------------------------
#include "../../GPUSort/benchmark/benchmarker.cpp"
#include "../../GPUSort/benchmark/measure.cu"