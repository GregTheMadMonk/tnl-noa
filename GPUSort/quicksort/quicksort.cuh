#pragma once

#include <TNL/Containers/Array.h>

__global__ void quicksortCuda(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> arr, int begin, int end)
{
    

}

void quicksort(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> arr)
{
    quicksortCuda<<<1, 1>>>(arr, 0, arr.getSize());
}
