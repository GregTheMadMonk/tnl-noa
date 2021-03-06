#pragma once

#include <TNL/Containers/Array.h>
#include "reduction.cuh"

#define deb(x) std::cout << #x << " = " << x << std::endl;

using CudaArrayView = TNL::Containers::ArrayView<int, TNL::Devices::Cuda>;

__device__ void cmpElem(CudaArrayView arr, int myBegin, int myEnd, int pivot, int &smaller, int &bigger)
{
    for (int i = myBegin + threadIdx.x; i < myEnd; i += blockDim.x)
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
                              int pivotIdx, int* newPivotIdx, int elemPerBlock, int * blockCount)
{
    static __shared__ int smallerStart, biggerStart;
    static __shared__ int pivot;

    const int myBegin = begin + elemPerBlock * blockIdx.x;
    const int myEnd = TNL::min(end - 1, myBegin + elemPerBlock); //important, pivot is at the end

    if(threadIdx.x == 0)
        pivot = arr[pivotIdx];
    __syncthreads();

    int smaller = 0, bigger = 0;
    cmpElem(arr, myBegin, myEnd, pivot, smaller, bigger);

    int smallerOffset = blockInclusivePrefixSum(smaller);
    int biggerOffset = blockInclusivePrefixSum(bigger);

    if (threadIdx.x == blockDim.x - 1)
    {
        smallerStart = atomicAdd(auxBeginIdx, smallerOffset);
        biggerStart = atomicAdd(auxEndIdx, -biggerOffset) - biggerOffset;
    }
    __syncthreads();

    copyData(arr, myBegin, myEnd, pivot, aux, smallerStart + smallerOffset - smaller, biggerStart + biggerOffset - bigger);

    if(threadIdx.x == 0)
    {
        if( atomicAdd(blockCount, -1) == 1)
        {
            *newPivotIdx = (*auxEndIdx) - 1;
            aux[*newPivotIdx] = pivot;
        }
    }
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
    TNL::Containers::Array<int, TNL::Devices::Cuda> aux(arr.getSize());
    TNL::Algorithms::MultiDeviceMemoryOperations<TNL::Devices::Cuda, TNL::Devices::Cuda >::
    copy(aux.getData(), arr.getData(), arr.getSize());
    
    TNL::Containers::Array<int, TNL::Devices::Cuda> helper({begin, end, 0, blocks});
    
    //------------------------------------
    
    cudaPartition<<<blocks, threadsPerBlock>>>(arr, begin, end,
        aux, helper.getData(), helper.getData() + 1,
        pivotIdx, helper.getData() + 2, elemPerBlock, helper.getData() + 3);
    
    //------------------------------------

    TNL::Algorithms::MultiDeviceMemoryOperations<TNL::Devices::Cuda, TNL::Devices::Cuda >::
    copy(arr.getData(), aux.getData(), aux.getSize());

    return helper.getElement(2);
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
