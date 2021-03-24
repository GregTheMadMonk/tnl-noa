#pragma once

#include <TNL/Containers/Array.h>
#include "cassert"
#include "../bitonicSort/bitonicSort.h"
#include "../util/reduction.cuh"
#include "cudaPartition.cuh"

using namespace TNL;
using namespace TNL::Containers;

template <typename Function, int externMemSize>
__device__ void externSort(ArrayView<int, TNL::Devices::Cuda> src,
                        ArrayView<int, TNL::Devices::Cuda> dst,
                        const Function & Cmp)
{
    static __shared__ int sharedMem[externMemSize];
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

template <typename Function, int stackSize>
__device__ void singleBlockQuickSort(ArrayView<int, TNL::Devices::Cuda> arr,
                                    ArrayView<int, TNL::Devices::Cuda> aux,
                                    const Function & Cmp, int _depth)
{
    static __shared__ int stackTop;
    static __shared__ int stackArrBegin[stackSize], stackArrEnd[stackSize], stackDepth[stackSize];
    static __shared__ int begin, end, depth,pivotBegin, pivotEnd;
    static __shared__ int pivot;

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
        if (threadIdx.x == 0)
        {
            begin = stackArrBegin[stackTop-1];
            end = stackArrEnd[stackTop-1];
            depth = stackDepth[stackTop-1];
            stackTop--;
            pivot = pickPivot((depth&1) == 0? 
                                    arr.getView(begin, end) :
                                    aux.getView(begin, end),
                                Cmp
                            );
        }
        __syncthreads();

        int size = end - begin;
        auto src = (depth&1) == 0 ? arr.getView(begin, end) : aux.getView(begin, end);
        auto dst = (depth&1) == 0 ? aux.getView(begin, end) : arr.getView(begin, end);

        if(size <= blockDim.x*2)
        {
            externSort<Function, 2048>(src, arr.getView(begin, end), Cmp);
            continue;
        }

        int smaller = 0, bigger = 0;
        countElem(src, smaller, bigger, pivot);

        int smallerOffset = blockInclusivePrefixSum(smaller);
        int biggerOffset = blockInclusivePrefixSum(bigger);

        if (threadIdx.x == blockDim.x - 1)
        {
            pivotBegin = smallerOffset;
            pivotEnd = size - biggerOffset;
        }
        __syncthreads();

        int destSmaller = 0 + smallerOffset - smaller;
        int destBigger = pivotEnd  + (biggerOffset - bigger);

        copyData(src, dst, destSmaller, destBigger, pivot);
        __syncthreads();

        for (int i = pivotBegin + threadIdx.x; i < pivotEnd; i += blockDim.x)
            src[i] = dst[i] = pivot;

        if(threadIdx.x == 0)
        {
            stackPush<stackSize>(stackArrBegin, stackArrEnd, stackDepth, stackTop,
                    begin, begin+ pivotBegin,
                    begin +pivotEnd, end,
                    depth);
        }
        __syncthreads();
    } //ends while loop
}