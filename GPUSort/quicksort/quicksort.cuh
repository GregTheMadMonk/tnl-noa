#pragma once

#include <TNL/Containers/Array.h>

void quicksort(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> arr, int begin, int end)
{
    if(begin >= end) return;

    int newPivotPos = partition(arr, begin, end, end-1);
    quicksort(arr, begin, newPivotPos);
    quicksort(arr, newPivotPos + 1, end);

    cudaDeviceSynchronize();
}

void quicksort(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> arr)
{
    quicksort(arr, 0, arr.getSize());
}
