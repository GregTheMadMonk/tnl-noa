/***************************************************************************
                          cudaPartition.h  -  description
                             -------------------
    begin                : Jul 13, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Xuan Thang Nguyen

#pragma once

#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/Sorting/detail/reduction.h>
#include <TNL/Algorithms/Sorting/detail/task.h>

#ifdef HAVE_CUDA

using namespace TNL;
using namespace TNL::Containers;

template <typename Value, typename Device, typename CMP>
__device__ Value pickPivot(TNL::Containers::ArrayView<Value, Device> src, const CMP &Cmp)
{
    //return src[0];
    //return src[src.getSize()-1];

    if (src.getSize() == 1)
        return src[0];

    const Value &a = src[0], &b = src[src.getSize() / 2], &c = src[src.getSize() - 1];

    if (Cmp(a, b)) // ..a..b..
    {
        if (Cmp(b, c)) // ..a..b..c
            return b;
        else if (Cmp(c, a)) //..c..a..b..
            return a;
        else //..a..c..b..
            return c;
    }
    else //..b..a..
    {
        if (Cmp(a, c)) //..b..a..c
            return a;
        else if (Cmp(c, b)) //..c..b..a..
            return b;
        else //..b..c..a..
            return c;
    }
}

template <typename Value, typename Device, typename CMP>
__device__ int pickPivotIdx(TNL::Containers::ArrayView<Value, Device> src, const CMP &Cmp)
{
    //return 0;
    //return src.getSize()-1;

    if (src.getSize() <= 1)
        return 0;

    const Value &a = src[0], &b = src[src.getSize() / 2], &c = src[src.getSize() - 1];

    if (Cmp(a, b)) // ..a..b..
    {
        if (Cmp(b, c)) // ..a..b..c
            return src.getSize() / 2;
        else if (Cmp(c, a)) //..c..a..b..
            return 0;
        else //..a..c..b..
            return src.getSize() - 1;
    }
    else //..b..a..
    {
        if (Cmp(a, c)) //..b..a..c
            return 0;
        else if (Cmp(c, b)) //..c..b..a..
            return src.getSize() / 2;
        else //..b..c..a..
            return src.getSize() - 1;
    }
}

//-----------------------------------------------------------

template <typename Value, typename CMP>
__device__ void countElem(ArrayView<Value, Devices::Cuda> arr,
                          const CMP &Cmp,
                          int &smaller, int &bigger,
                          const Value &pivot)
{
    for (int i = threadIdx.x; i < arr.getSize(); i += blockDim.x)
    {
        const Value &data = arr[i];
        if (Cmp(data, pivot))
            smaller++;
        else if (Cmp(pivot, data))
            bigger++;
    }
}

//-----------------------------------------------------------

template <typename Value, typename CMP>
__device__ void copyDataShared(ArrayView<Value, Devices::Cuda> src,
                               ArrayView<Value, Devices::Cuda> dst,
                               const CMP &Cmp,
                               Value *sharedMem,
                               int smallerStart, int biggerStart,
                               int smallerTotal, int biggerTotal,
                               int smallerOffset, int biggerOffset, //exclusive prefix sum of elements
                               const Value &pivot)
{

    for (int i = threadIdx.x; i < src.getSize(); i += blockDim.x)
    {
        const Value &data = src[i];
        if (Cmp(data, pivot))
            sharedMem[smallerOffset++] = data;
        else if (Cmp(pivot, data))
            sharedMem[smallerTotal + biggerOffset++] = data;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < smallerTotal + biggerTotal; i += blockDim.x)
    {
        if (i < smallerTotal)
            dst[smallerStart + i] = sharedMem[i];
        else
            dst[biggerStart + i - smallerTotal] = sharedMem[i];
    }
}

template <typename Value, typename CMP>
__device__ void copyData(ArrayView<Value, Devices::Cuda> src,
                         ArrayView<Value, Devices::Cuda> dst,
                         const CMP &Cmp,
                         int smallerStart, int biggerStart,
                         const Value &pivot)
{
    for (int i = threadIdx.x; i < src.getSize(); i += blockDim.x)
    {
        const Value &data = src[i];
        if (Cmp(data, pivot))
        {
            /*
            if(smallerStart >= dst.getSize() || smallerStart < 0)
                printf("failed smaller: b:%d t:%d: tried to write into [%d]/%d\n", blockDim.x, threadIdx.x, smallerStart, dst.getSize());
            */
            dst[smallerStart++] = data;
        }
        else if (Cmp(pivot, data))
        {
            /*
            if(biggerStart >= dst.getSize() || biggerStart < 0)
                printf("failed bigger: b:%d t:%d: tried to write into [%d]/%d\n", blockDim.x, threadIdx.x, biggerStart, dst.getSize());
            */
            dst[biggerStart++] = data;
        }
    }
}

//----------------------------------------------------------------------------------

template <typename Value, typename CMP, bool useShared>
__device__ void cudaPartition(ArrayView<Value, Devices::Cuda> src,
                              ArrayView<Value, Devices::Cuda> dst,
                              const CMP &Cmp,
                              Value *sharedMem,
                              const Value &pivot,
                              int elemPerBlock, TASK &task)
{
    static __shared__ int smallerStart, biggerStart;

    int myBegin = elemPerBlock * (blockIdx.x - task.firstBlock);
    int myEnd = TNL::min(myBegin + elemPerBlock, src.getSize());

    auto srcView = src.getView(myBegin, myEnd);

    //-------------------------------------------------------------------------

    int smaller = 0, bigger = 0;
    countElem(srcView, Cmp, smaller, bigger, pivot);

    int smallerPrefSumInc = blockInclusivePrefixSum(smaller);
    int biggerPrefSumInc = blockInclusivePrefixSum(bigger);

    if (threadIdx.x == blockDim.x - 1) //last thread in block has sum of all values
    {
        smallerStart = atomicAdd(&(task.dstBegin), smallerPrefSumInc);
        biggerStart = atomicAdd(&(task.dstEnd), -biggerPrefSumInc) - biggerPrefSumInc;
    }
    __syncthreads();

    //-----------------------------------------------------------
    if (useShared)
    {
        static __shared__ int smallerTotal, biggerTotal;
        if (threadIdx.x == blockDim.x - 1)
        {
            smallerTotal = smallerPrefSumInc;
            biggerTotal = biggerPrefSumInc;
        }
        __syncthreads();

        copyDataShared(srcView, dst, Cmp, sharedMem,
                       smallerStart, biggerStart,
                       smallerTotal, biggerTotal,
                       smallerPrefSumInc - smaller, biggerPrefSumInc - bigger, //exclusive prefix sum of elements
                       pivot);
    }
    else
    {
        int destSmaller = smallerStart + smallerPrefSumInc - smaller;
        int destBigger = biggerStart + biggerPrefSumInc - bigger;
        copyData(srcView, dst, Cmp, destSmaller, destBigger, pivot);
    }
}

#endif
