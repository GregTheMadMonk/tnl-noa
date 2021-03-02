#pragma once

#include <TNL/Containers/Array.h>

__global__ void quicksortCuda(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> arr, int begin, int end)
{
    if(begin >= end)
        return;

    int pivotIdx = end - 1;
    int pivot = arr[pivotIdx];

    int midPoint = begin; //[begin ; midPoint) contain elems smaller than pivot

    //partition the array except for last elem (the pivot itself) 
    for(int i = begin; i + 1< end; i++)
    {
        if(arr[i] < pivot)
        {
            TNL::swap(arr[i], arr[midPoint]);
            midPoint++; //increase boundary
        }
    }

    //put pivot onto its correct position, now [begin, midpoint] is sorted
    TNL::swap(arr[midPoint], arr[pivotIdx]);

    //sorts all elems before midPoint(which is pivot now)
    quicksortCuda<<<1, 1>>>(arr, begin, midPoint);

    //sorts all elems after(bigger than) midPoint
    quicksortCuda<<<1, 1>>>(arr, midPoint+1, end);

}

void quicksort(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> arr)
{
    quicksortCuda<<<1, 1>>>(arr, 0, arr.getSize());
}
