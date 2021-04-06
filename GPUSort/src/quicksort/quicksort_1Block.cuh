#pragma once

#include <TNL/Containers/Array.h>
#include "cassert"
#include "../bitonicSort/bitonicSort.h"
#include "../util/reduction.cuh"
#include "cudaPartition.cuh"

using namespace TNL;
using namespace TNL::Containers;

template <typename Value, typename Function, int externMemSize>
__device__ void externSort(ArrayView<Value, TNL::Devices::Cuda> src,
                        ArrayView<Value, TNL::Devices::Cuda> dst,
                        const Function & Cmp)
{
    static __shared__ Value sharedMem[externMemSize];
    bitonicSort_Block(src, dst, sharedMem, Cmp);
}

template<int stackSize>
__device__ void stackPush(int stackArrBegin[], int stackArrEnd[],
                            int stackDepth[], int & stackTop,
                            int begin, int pivotBegin,
                            int pivotEnd, int end,
                            int depth)
{
    int sizeL = pivotBegin - begin, sizeR = end - pivotEnd;
    
    //push the bigger one 1st and then smaller one 2nd
    //in next iteration, the smaller part will be handled 1st
    if(sizeL > sizeR)
    {
        if(sizeL > 0) //left from pivot are smaller elems
        {
            stackArrBegin[stackTop] = begin;
            stackArrEnd[stackTop] = pivotBegin;
            stackDepth[stackTop] = depth + 1;
            stackTop++;
        }
        
        if(sizeR > 0) //right from pivot until end are elem greater than pivot
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
        if(sizeR > 0) //right from pivot until end are elem greater than pivot
        {
            stackArrBegin[stackTop] = pivotEnd;
            stackArrEnd[stackTop] = end;
            stackDepth[stackTop] = depth + 1;
            stackTop++;
        }

        if(sizeL > 0) //left from pivot are smaller elems
        {
            assert(stackTop < stackSize && "Local quicksort stack overflow.");

            stackArrBegin[stackTop] = begin;
            stackArrEnd[stackTop] = pivotBegin;
            stackDepth[stackTop] = depth + 1;
            stackTop++;
        }
    }
}

template <typename Value, typename Function, int stackSize>
__device__ void singleBlockQuickSort(ArrayView<Value, TNL::Devices::Cuda> arr,
                                    ArrayView<Value, TNL::Devices::Cuda> aux,
                                    const Function & Cmp, int _depth)
{
    if(arr.getSize() <= blockDim.x*2)
    {
        auto src = (_depth &1) == 0? arr : aux;
        externSort<Value, Function, 2048>(src, arr, Cmp);
        return;
    }

    static __shared__ int stackTop;
    static __shared__ int stackArrBegin[stackSize], stackArrEnd[stackSize], stackDepth[stackSize];
    static __shared__ int begin, end, depth;
    static __shared__ int pivotBegin, pivotEnd;
    static __shared__ Value pivot;

    if (threadIdx.x == 0)
    {
        stackTop = 0;
        stackArrBegin[stackTop] = 0;
        stackArrEnd[stackTop] = arr.getSize();
        stackDepth[stackTop] = _depth;
        stackTop++;
    }
    __syncthreads();

    while(stackTop > 0)
    {
        //pick up partition to break up
        if (threadIdx.x == 0)
        {
            begin = stackArrBegin[stackTop-1];
            end = stackArrEnd[stackTop-1];
            depth = stackDepth[stackTop-1];
            stackTop--;
        }
        __syncthreads();

        int size = end - begin;
        auto &src = (depth&1) == 0 ? arr : aux;

        //small enough for for bitonic
        if(size <= blockDim.x*2)
        {
            externSort<Value, Function, 2048>(src.getView(begin, end), arr.getView(begin, end), Cmp);
            __syncthreads();
            continue;
        }
        //------------------------------------------------------

        //actually do partitioning from here on out
        if(threadIdx.x == 0)
            pivot = pickPivot(src.getView(begin, end),Cmp);
        __syncthreads();

        int smaller = 0, bigger = 0;
        countElem(src.getView(begin, end), smaller, bigger, pivot);

        //synchronization is in this function already
        int smallerOffset = blockInclusivePrefixSum(smaller);
        int biggerOffset = blockInclusivePrefixSum(bigger);

        if (threadIdx.x == blockDim.x - 1) //has sum of all smaller and greater elements than pivot in src
        {
            pivotBegin = 0 + smallerOffset;
            pivotEnd = size - biggerOffset;
        }
        __syncthreads();

        int destSmaller = 0 + (smallerOffset - smaller);
        int destBigger = pivotEnd  + (biggerOffset - bigger);
        auto &dst = (depth&1) == 0 ? aux : arr;

        copyData(src.getView(begin, end), dst.getView(begin, end), destSmaller, destBigger, pivot);
        __syncthreads();

        for (int i = pivotBegin + threadIdx.x; i < pivotEnd; i += blockDim.x)
            arr[begin + i] = pivot;

        //creates new tasks
        if(threadIdx.x == 0)
        {
            stackPush<stackSize>(stackArrBegin, stackArrEnd, stackDepth, stackTop,
                    begin, begin+ pivotBegin,
                    begin +pivotEnd, end,
                    depth);
        }
        __syncthreads(); //sync to update stackTop
    } //ends while loop
}