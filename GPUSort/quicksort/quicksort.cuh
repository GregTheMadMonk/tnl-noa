#pragma once

#include <TNL/Containers/Array.h>

__device__ void cmpElem(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> arr, int myBegin, int myEnd,
    int pivot, int &smaller, int&bigger)
{
    for(int i = myBegin + threadIdx.x; i < myEnd; i+= threadIdx.x)
    {
        int data = arr[i];
        if(data < pivot) smaller++;
        else bigger++;
    }
}

__global__ void cudaPartition(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> arr,
                                int begin, int end, int pivotIdx, int * newPivotPos,
                                int elemPerBlock)
{
    const int myBegin = begin + elemPerBlock*blockIdx.x;
    const int myEnd = TNL::min(end - 1, myBegin + elemPerBlock); //important, pivot is at the end

    int pivot = arr[pivotIdx];
    int smaller = 0, bigger = 0;
    cmpElem(arr, myBegin, myEnd, pivot, smaller, bigger);    



}

int partition(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> arr, int begin, int end, int pivotIdx)
{
    int size = end - begin;
    const int threadsPerBlock = 512, maxBlocks = 1<<14; //16k
    int elemPerBlock, blocks;
    
    int setsNeeded = size/threadsPerBlock + (size % threadsPerBlock != 0);
    if(setsNeeded <= maxBlocks)
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
    TNL::Containers::Array<int, TNL::Devices::Cuda> newPivotPos;
    cudaPartition<<<blocks, maxBlocks>>>(arr, begin, end, pivotIdx, newPivotPos.getData(), elemPerBlock);

    return newPivotPos.getElement(0);
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
