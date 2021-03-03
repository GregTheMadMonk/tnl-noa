#pragma once

#include <TNL/Containers/Array.h>
#include "reduction.cuh"

using CudaArrayView = TNL::Containers::ArrayView<int, TNL::Devices::Cuda>;

__device__ void cmpElem(CudaArrayView arr, int myBegin, int myEnd, int pivot, int &smaller, int &bigger)
{
    for (int i = myBegin + threadIdx.x; i < myEnd; i += threadIdx.x)
    {
        int data = arr[i];
        if (data < pivot)
            smaller++;
        else
            bigger++;
    }
}

__device__ void copyData(CudaArrayView arr, int myBegin, int myEnd, int pivot,
                         CudaArrayView aux, int smallerStart, int biggerStart)
{
    for (int i = myBegin + threadIdx.x; i < myEnd; i += blockDim.x)
    {
        int data = arr[i];
        if (data < pivot)
            aux[smallerStart++] = data;
        else
            aux[biggerStart++] = data;
    }
}

__global__ void cudaPartition(CudaArrayView arr, int begin, int end,
                              CudaArrayView aux, int *auxBeginIdx, int *auxEndIdx,
                              int pivotIdx, int *newPivotPos,
                              int elemPerBlock)
{
    static __shared__ int sharedMem[2];
    int *smallerStart = sharedMem, *biggerStart = smallerStart + 1;

    const int myBegin = begin + elemPerBlock * blockIdx.x;
    const int myEnd = TNL::min(end - 1, myBegin + elemPerBlock); //important, pivot is at the end

    int pivot = arr[pivotIdx];
    int smaller = 0, bigger = 0;
    cmpElem(arr, myBegin, myEnd, pivot, smaller, bigger);

    int smallerOffset = blockReduceSum(smaller);
    int biggerOffset = blockReduceSum(bigger);

    if (threadIdx.x == 0)
    {
        *smallerStart = atomicAdd(auxBeginIdx, smallerOffset);
        *biggerStart = atomicAdd(auxEndIdx, -biggerOffset) - biggerOffset;
    }
    __syncthreads();

    int auxThreadSmallerBegin = atomicAdd(smallerStart, smaller);
    int auxThreadBiggerBegin = atomicAdd(biggerStart, bigger);
    copyData(arr, myBegin, myEnd, pivot, aux, auxThreadSmallerBegin, auxThreadBiggerBegin);
}

int partition(CudaArrayView arr, int begin, int end, int pivotIdx)
{
    int size = end - begin;
    const int threadsPerBlock = 512, maxBlocks = 1 << 14; //16k
    int elemPerBlock, blocks;

    int setsNeeded = size / threadsPerBlock + (size % threadsPerBlock != 0);
    if (setsNeeded <= maxBlocks)
    {
        blocks = setsNeeded;
        elemPerBlock = threadsPerBlock;
    }
    else
    {
        int setsPerBlock = setsNeeded / blocks + 1; //+1 to spread out task of the last block
        elemPerBlock *= setsPerBlock;
        blocks = size / elemPerBlock + (size % elemPerBlock != 0);
    }

    //------------------------------------
    TNL::Containers::Array<int, TNL::Devices::Cuda> aux(end - begin), cudaAuxBegin({0}), cudaAuxEnd({end});
    TNL::Containers::Array<int, TNL::Devices::Cuda> newPivotPos;
    cudaPartition<<<blocks, maxBlocks>>>(arr, begin, end,
                                         aux, cudaAuxBegin.getData(), cudaAuxEnd.getData(),
                                         pivotIdx, newPivotPos.getData(),
                                         elemPerBlock);

    return newPivotPos.getElement(0);
}

//-----------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------

void quicksort(CudaArrayView arr, int begin, int end)
{
    if (begin >= end)
        return;

    int newPivotPos = partition(arr, begin, end, end - 1);
    quicksort(arr, begin, newPivotPos);
    quicksort(arr, newPivotPos + 1, end);

    cudaDeviceSynchronize();
}

void quicksort(CudaArrayView arr)
{
    quicksort(arr, 0, arr.getSize());
}
