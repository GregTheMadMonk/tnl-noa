#pragma once

#include <TNL/Containers/Array.h>
#include "cassert"
#include <TNL/Algorithms/detail/Sorting/bitonicSort.h>
#include <TNL/Algorithms/detail/Sorting/reduction.cuh>
#include <TNL/Algorithms/detail/Sorting/cudaPartition.cuh>

using namespace TNL;
using namespace TNL::Containers;

template <typename Value, typename CMP>
__device__ void externSort(ArrayView<Value, TNL::Devices::Cuda> src,
                           ArrayView<Value, TNL::Devices::Cuda> dst,
                           const CMP &Cmp, Value *sharedMem)
{
    bitonicSort_Block(src, dst, sharedMem, Cmp);
}

template <typename Value, typename CMP>
__device__ void externSort(ArrayView<Value, TNL::Devices::Cuda> src,
                           const CMP &Cmp)
{
    bitonicSort_Block(src, Cmp);
}

//---------------------------------------------------------------

template <int stackSize>
__device__ void stackPush(int stackArrBegin[], int stackArrEnd[],
                          int stackDepth[], int &stackTop,
                          int begin, int pivotBegin,
                          int pivotEnd, int end,
                          int iteration);

//---------------------------------------------------------------

template <typename Value, typename CMP, int stackSize, bool useShared>
__device__ void singleBlockQuickSort(ArrayView<Value, TNL::Devices::Cuda> arr,
                                     ArrayView<Value, TNL::Devices::Cuda> aux,
                                     const CMP &Cmp, int _iteration,
                                     Value *sharedMem, int memSize,
                                     int maxBitonicSize)
{
    if (arr.getSize() <= maxBitonicSize)
    {
        auto &src = (_iteration & 1) == 0 ? arr : aux;
        if (useShared && arr.getSize() <= memSize)
            externSort<Value, CMP>(src, arr, Cmp, sharedMem);
        else
        {
            externSort<Value, CMP>(src, Cmp);
            //extern sort without shared memory only works in-place, need to copy into from aux
            if ((_iteration & 1) != 0)
                for (int i = threadIdx.x; i < arr.getSize(); i += blockDim.x)
                    arr[i] = src[i];
        }

        return;
    }

    static __shared__ int stackTop;
    static __shared__ int stackArrBegin[stackSize], stackArrEnd[stackSize], stackDepth[stackSize];
    static __shared__ int begin, end, iteration;
    static __shared__ int pivotBegin, pivotEnd;
    Value *piv = sharedMem;
    sharedMem += 1;

    if (threadIdx.x == 0)
    {
        stackTop = 0;
        stackArrBegin[stackTop] = 0;
        stackArrEnd[stackTop] = arr.getSize();
        stackDepth[stackTop] = _iteration;
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
            iteration = stackDepth[stackTop - 1];
            stackTop--;
        }
        __syncthreads();

        int size = end - begin;
        auto &src = (iteration & 1) == 0 ? arr : aux;

        //small enough for for bitonic
        if (size <= maxBitonicSize)
        {
            if (useShared && size <= memSize)
                externSort<Value, CMP>(src.getView(begin, end), arr.getView(begin, end), Cmp, sharedMem);
            else
            {
                externSort<Value, CMP>(src.getView(begin, end), Cmp);
                //extern sort without shared memory only works in-place, need to copy into from aux
                if ((iteration & 1) != 0)
                    for (int i = threadIdx.x; i < src.getSize(); i += blockDim.x)
                        arr[begin + i] = src[i];
            }
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

        auto &dst = (iteration & 1) == 0 ? aux : arr;

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
                                 iteration);
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
                          int iteration)
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
            stackDepth[stackTop] = iteration + 1;
            stackTop++;
        }

        if (sizeR > 0) //right from pivot until end are elem greater than pivot
        {
            assert(stackTop < stackSize && "Local quicksort stack overflow.");

            stackArrBegin[stackTop] = pivotEnd;
            stackArrEnd[stackTop] = end;
            stackDepth[stackTop] = iteration + 1;
            stackTop++;
        }
    }
    else
    {
        if (sizeR > 0) //right from pivot until end are elem greater than pivot
        {
            stackArrBegin[stackTop] = pivotEnd;
            stackArrEnd[stackTop] = end;
            stackDepth[stackTop] = iteration + 1;
            stackTop++;
        }

        if (sizeL > 0) //left from pivot are smaller elems
        {
            assert(stackTop < stackSize && "Local quicksort stack overflow.");

            stackArrBegin[stackTop] = begin;
            stackArrEnd[stackTop] = pivotBegin;
            stackDepth[stackTop] = iteration + 1;
            stackTop++;
        }
    }
}