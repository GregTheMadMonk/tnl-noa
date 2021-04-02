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
    //return src[0];
    //return src[src.getSize()-1];

    if(src.getSize() ==1)
        return src[0];
    
    Value a = src[0], b = src[src.getSize()/2], c = src[src.getSize() - 1];

    if(Cmp(a, b)) // ..a..b..
    {
        if(Cmp(b, c))// ..a..b..c
            return b;
        else if(Cmp(c, a))//..c..a..b..
            return a;
        else //..a..c..b..
            return c;
    }
    else //..b..a..
    {
        if(Cmp(a, c))//..b..a..c
            return a;
        else if(Cmp(c, b))//..c..b..a..
            return b;
        else //..b..c..a..
            return c;
    }
    
}

template <typename Value, typename Device, typename Function>
__device__ Value pickPivotIdx(TNL::Containers::ArrayView<Value, Device> src, const Function & Cmp)
{
    //return 0;
    //return src.getSize()-1;

    if(src.getSize() <= 1)
        return 0;
    
    Value a = src[0], b = src[src.getSize()/2], c = src[src.getSize() - 1];

    if(Cmp(a, b)) // ..a..b..
    {
        if(Cmp(b, c))// ..a..b..c
            return src.getSize()/2;
        else if(Cmp(c, a))//..c..a..b..
            return 0;
        else //..a..c..b..
            return src.getSize() - 1;
    }
    else //..b..a..
    {
        if(Cmp(a, c))//..b..a..c
            return 0;
        else if(Cmp(c, b))//..c..b..a..
            return src.getSize()/2;
        else //..b..c..a..
            return src.getSize() - 1;
    }
}

__device__
void countElem(ArrayView<int, Devices::Cuda> arr,
             int &smaller, int &bigger,
             const int &pivot)
{
    for (int i = threadIdx.x; i < arr.getSize(); i += blockDim.x)
    {
        int data = arr[i];
        smaller += (data < pivot);
        bigger += (data > pivot);
    }
}

__device__
void copyDataShared(ArrayView<int, Devices::Cuda> src,
              ArrayView<int, Devices::Cuda> dst,
              int *sharedMem,
              int smallerStart, int biggerStart,
              int smallerTotal, int biggerTotal,
              int smallerOffset, int biggerOffset, //exclusive prefix sum of elements
              const int &pivot)
{

    for (int i = threadIdx.x; i < src.getSize(); i += blockDim.x)
    {
        int data = src[i];
        if (data < pivot)
            sharedMem[smallerOffset++] = data;
        else if (data > pivot)
            sharedMem[smallerTotal + biggerOffset++] = data;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < smallerTotal + biggerTotal; i += blockDim.x)
    {
        if(i < smallerTotal)
            dst[smallerStart + i] = sharedMem[i];
        else
            dst[biggerStart + i - smallerTotal] = sharedMem[i];
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
        {
            /*
            if(smallerStart >= dst.getSize() || smallerStart < 0)
                printf("failed here: b:%d t:%d: tried to write into [%d]/%d\n", blockDim.x, threadIdx.x, smallerStart, dst.getSize());
            */
            dst[smallerStart++] = data;
        }
        else if (data > pivot)
        {
            
            if(biggerStart >= dst.getSize() || biggerStart < 0)
                printf("failed here: b:%d t:%d: tried to write into [%d]/%d\n", blockDim.x, threadIdx.x, biggerStart, dst.getSize());
            
            dst[biggerStart++] = data;
        }
    }
}

//----------------------------------------------------------------------------------

template <typename Function>
__device__ void cudaPartition(ArrayView<int, Devices::Cuda> src,
                              ArrayView<int, Devices::Cuda> dst,
                              int * sharedMem,
                              const Function &Cmp, const int & pivot,
                              int elemPerBlock, TASK & task
                              )
{
    static __shared__ int smallerStart, biggerStart;
    static __shared__ int smallerTotal, biggerTotal;

    int myBegin = elemPerBlock * (blockIdx.x - task.firstBlock);
    int myEnd = TNL::min(myBegin + elemPerBlock, src.getSize());

    auto srcView = src.getView(myBegin, myEnd);

    //-------------------------------------------------------------------------

    int smaller = 0, bigger = 0;
    countElem(srcView, smaller, bigger, pivot);

    int smallerPrefSumInc = blockInclusivePrefixSum(smaller);
    int biggerPrefSumInc = blockInclusivePrefixSum(bigger);

    if (threadIdx.x == blockDim.x - 1) //last thread in block has sum of all values
    {
        smallerStart = atomicAdd(&(task.dstBegin), smallerPrefSumInc);
        biggerStart = atomicAdd(&(task.dstEnd), -biggerPrefSumInc) - biggerPrefSumInc;
        smallerTotal = smallerPrefSumInc;
        biggerTotal = biggerPrefSumInc;
    }
    __syncthreads();

    //-----------------------------------------------------------

    /*
    int destSmaller = smallerStart + smallerPrefSumInc - smaller;
    int destBigger = biggerStart + biggerPrefSumInc - bigger;
    copyData(srcView, dst, destSmaller, destBigger, pivot);
    */

    copyDataShared(srcView, dst, sharedMem,
                    smallerStart, biggerStart,
                    smallerTotal, biggerTotal,
                    smallerPrefSumInc - smaller, biggerPrefSumInc - bigger, //exclusive prefix sum of elements
                    pivot);
}