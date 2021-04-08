#pragma once

#include <TNL/Containers/Array.h>
#include "cassert"
#include "../bitonicSort/bitonicSort.h"
#include "../util/reduction.cuh"
#include "cudaPartition.cuh"

using namespace TNL;
using namespace TNL::Containers;

template <typename Value, typename Function>
__device__ void externSort(ArrayView<Value, TNL::Devices::Cuda> src,
                           ArrayView<Value, TNL::Devices::Cuda> dst,
                           const Function &Cmp, Value *sharedMem)
{
    bitonicSort_Block(src, dst, sharedMem, Cmp);
}

template <typename Value, typename Function>
__device__ void externSort(ArrayView<Value, TNL::Devices::Cuda> src,
                           ArrayView<Value, TNL::Devices::Cuda> dst,
                           const Function &Cmp)
{
    bitonicSort_Block(src, dst, Cmp);
}

//---------------------------------------------------------------

template <int stackSize>
__device__ void stackPush(int stackArrBegin[], int stackArrEnd[],
                          int stackDepth[], int &stackTop,
                          int begin, int pivotBegin,
                          int pivotEnd, int end,
                          int depth);

//---------------------------------------------------------------

template <typename Value, typename Function, int stackSize, bool useShared>
__device__ void singleBlockQuickSort(ArrayView<Value, TNL::Devices::Cuda> arr,
                                     ArrayView<Value, TNL::Devices::Cuda> aux,
                                     const Function &Cmp, int _depth,
                                     Value *sharedMem, int memSize)
{
    if (arr.getSize() <= blockDim.x * 2)
    {
        auto &src = (_depth & 1) == 0 ? arr : aux;
        if (useShared && arr.getSize() <= memSize)
            externSort<Value, Function>(src, arr, Cmp, sharedMem);
        else
            externSort<Value, Function>(src, arr, Cmp);

        return;
    }

    static __shared__ int stackTop;
    static __shared__ int stackArrBegin[stackSize], stackArrEnd[stackSize], stackDepth[stackSize];
    static __shared__ int begin, end, depth;
    static __shared__ int pivotBegin, pivotEnd;
    Value *piv = sharedMem;
    sharedMem += 1;

    if (threadIdx.x == 0)
    {
        stackTop = 0;
        stackArrBegin[stackTop] = 0;
        stackArrEnd[stackTop] = arr.getSize();
        stackDepth[stackTop] = _depth;
        stackTop++;
    }
    __syncthreads();

    while (stackTop > 0)
    {
        //pick up partition to break up
        if (threadIdx.x == 0)
        {
            begin = stackArrBegin[stackTop - 1];
            end = stackArrEnd[stackTop - 1];
            depth = stackDepth[stackTop - 1];
            stackTop--;
        }
        __syncthreads();

        int size = end - begin;
        auto &src = (depth & 1) == 0 ? arr : aux;

        //small enough for for bitonic
        if (size <= blockDim.x * 2)
        {
            if (useShared && size <= memSize)
                externSort<Value, Function>(src.getView(begin, end), arr.getView(begin, end), Cmp, sharedMem);
            else
                externSort<Value, Function>(src.getView(begin, end), arr.getView(begin, end), Cmp);
            __syncthreads();
            continue;
        }

        //------------------------------------------------------

        if (threadIdx.x == 0)
            *piv = pickPivot(src.getView(begin, end), Cmp);
        __syncthreads();
        Value &pivot = *piv;

        int smaller = 0, bigger = 0;
        countElem(src.getView(begin, end), Cmp, smaller, bigger, pivot);

        //synchronization is in this function already
        int smallerPrefSumInc = blockInclusivePrefixSum(smaller);
        int biggerPrefSumInc = blockInclusivePrefixSum(bigger);

        if (threadIdx.x == blockDim.x - 1) //has sum of all smaller and greater elements than pivot in src
        {
            pivotBegin = 0 + smallerPrefSumInc;
            pivotEnd = size - biggerPrefSumInc;
        }
        __syncthreads();

        //--------------------------------------------------------------
        /**
         * move elements, either use shared mem for coalesced access or without shared mem if data is too big
         * */

        auto &dst = (depth & 1) == 0 ? aux : arr;

        if (useShared && size <= memSize)
        {
            static __shared__ int smallerTotal, biggerTotal;
            if (threadIdx.x == blockDim.x - 1)
            {
                smallerTotal = smallerPrefSumInc;
                biggerTotal = biggerPrefSumInc;
            }
            __syncthreads();

            copyDataShared(src.getView(begin, end), dst.getView(begin, end),
                           Cmp, sharedMem,
                           0, pivotEnd,
                           smallerTotal, biggerTotal,
                           smallerPrefSumInc - smaller, biggerPrefSumInc - bigger, //exclusive prefix sum of elements
                           pivot);
        }
        else
        {
            int destSmaller = 0 + (smallerPrefSumInc - smaller);
            int destBigger = pivotEnd + (biggerPrefSumInc - bigger);

            copyData(src.getView(begin, end), dst.getView(begin, end), Cmp, destSmaller, destBigger, pivot);
        }

        __syncthreads();

        for (int i = pivotBegin + threadIdx.x; i < pivotEnd; i += blockDim.x)
            arr[begin + i] = pivot;

        //creates new tasks
        if (threadIdx.x == 0)
        {
            stackPush<stackSize>(stackArrBegin, stackArrEnd, stackDepth, stackTop,
                                 begin, begin + pivotBegin,
                                 begin + pivotEnd, end,
                                 depth);
        }
        __syncthreads(); //sync to update stackTop
    }                    //ends while loop
}

//--------------------------------------------------------------

template <int stackSize>
__device__ void stackPush(int stackArrBegin[], int stackArrEnd[],
                          int stackDepth[], int &stackTop,
                          int begin, int pivotBegin,
                          int pivotEnd, int end,
                          int depth)
{
    int sizeL = pivotBegin - begin, sizeR = end - pivotEnd;

    //push the bigger one 1st and then smaller one 2nd
    //in next iteration, the smaller part will be handled 1st
    if (sizeL > sizeR)
    {
        if (sizeL > 0) //left from pivot are smaller elems
        {
            stackArrBegin[stackTop] = begin;
            stackArrEnd[stackTop] = pivotBegin;
            stackDepth[stackTop] = depth + 1;
            stackTop++;
        }

        if (sizeR > 0) //right from pivot until end are elem greater than pivot
        {
            assert(stackTop < stackSize && "Local quicksort stack overflow.");

            stackArrBegin[stackTop] = pivotEnd;
            stackArrEnd[stackTop] = end;
            stackDepth[stackTop] = depth + 1;
            stackTop++;
        }
    }
    else
    {
        if (sizeR > 0) //right from pivot until end are elem greater than pivot
        {
            stackArrBegin[stackTop] = pivotEnd;
            stackArrEnd[stackTop] = end;
            stackDepth[stackTop] = depth + 1;
            stackTop++;
        }

        if (sizeL > 0) //left from pivot are smaller elems
        {
            assert(stackTop < stackSize && "Local quicksort stack overflow.");

            stackArrBegin[stackTop] = begin;
            stackArrEnd[stackTop] = pivotBegin;
            stackDepth[stackTop] = depth + 1;
            stackTop++;
        }
    }
}