#pragma once

#include <TNL/Containers/Array.h>

int partition(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> arr, int begin, int end, int pivotIdx)
{
    int size = end - begin;
    const int threadsPerBlock = 512, maxBlocks = 1<<14; //16k
    int elemPerBlock, blocks;
    
    int setsNeeded = size/threadsPerBlock + (size % threadsPerBlock != 0);
    if(setsNeeded <= blocks)
    {
        blocks = setsNeeded;
        elemPerBlock = threadsPerBlock;
    }
    else
    {
        int setsPerBlock = setsNeeded/blocks + 1; //+1 to spread out task of the last block
        elemPerBlock *= setsPerBlock;
        blocks = size / elemPerBlock + (size % elemPerBlock != 0);
    }

    //------------------------------------



}

//-----------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------

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
