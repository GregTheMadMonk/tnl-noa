#pragma once

#include <TNL/Containers/Array.h>
#include "../util/reduction.cuh"
#include "task.h"
#include <iostream>

#define deb(x) std::cout << #x << " = " << x << std::endl;

using namespace TNL;
using namespace TNL::Containers;

template <typename Value, typename Device, typename Function>
__device__ Value pickPivot(TNL::Containers::ArrayView<Value, Device> src, const Function & Cmp)
{
    return src[0];
}

__device__
void countElem(ArrayView<int, Devices::Cuda> arr,
             int &smaller, int &bigger,
             const int &pivot)
{
    for (int i = threadIdx.x; i < arr.getSize(); i += blockDim.x)
    {
        int data = arr[i];
        if (data < pivot)
            smaller++;
        else if (data > pivot)
            bigger++;
    }
}

__device__
void copyData(ArrayView<int, Devices::Cuda> src,
              ArrayView<int, Devices::Cuda> dst,
              int smallerStart, int biggerStart,
              const int &pivot)
{
    for (int i = threadIdx.x; i < src.getSize(); i += blockDim.x)
    {
        int data = src[i];
        if (data < pivot)
            dst[smallerStart++] = data;
        else if (data > pivot)
            dst[biggerStart++] = data;
    }
}

//----------------------------------------------------------------------------------

template <typename Function>
__device__ bool cudaPartition(ArrayView<int, Devices::Cuda> src,
                              ArrayView<int, Devices::Cuda> dst,
                              const Function &Cmp, const int & pivot,
                              int elemPerBlock, TASK & task
                              )
{
    static __shared__ int myBegin, myEnd;
    static __shared__ int smallerStart, biggerStart;
    static __shared__ bool writePivot;

    if (threadIdx.x == 0)
    {
        myBegin = elemPerBlock * (blockIdx.x - task.firstBlock);
        myEnd = TNL::min(myBegin + elemPerBlock, src.getSize());
    }
    __syncthreads();

    auto srcView = src.getView(myBegin, myEnd);

    //-------------------------------------------------------------------------

    int smaller = 0, bigger = 0;
    countElem(srcView, smaller, bigger, pivot);

    int smallerOffset = blockInclusivePrefixSum(smaller);
    int biggerOffset = blockInclusivePrefixSum(bigger);

    if (threadIdx.x == blockDim.x - 1) //last thread in block has sum of all values
    {
        smallerStart = atomicAdd(&(task.dstBegin), smallerOffset);
        biggerStart = atomicAdd(&(task.dstEnd), -biggerOffset) - biggerOffset;
    }
    __syncthreads();

    //-----------------------------------------------------------

    int destSmaller = smallerStart + smallerOffset - smaller;
    int destBigger = biggerStart + biggerOffset - bigger;
    copyData(srcView, dst, destSmaller, destBigger, pivot);
    __syncthreads();

    //-----------------------------------------------------------

    if (threadIdx.x == 0)
        writePivot = atomicAdd(&(task.stillWorkingCnt), -1) == 1;
    __syncthreads();

    return writePivot;
}