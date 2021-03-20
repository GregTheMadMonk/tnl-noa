#include "quicksort.cuh"

#include <TNL/Containers/Array.h>
#include "reduction.cuh"
#include "task.h"
#include "../bitonicSort/bitonicSort.h"
#include "helper.cuh"
#include <iostream>
#include <cassert>
#include <cmath>

#define deb(x) std::cout << #x << " = " << x << std::endl;

using CudaArrayView = TNL::Containers::ArrayView<int, TNL::Devices::Cuda>;
using CudaTaskArray = TNL::Containers::Array<TASK, TNL::Devices::Cuda>;

template <typename Function>
__device__ bool cudaPartition(CudaArrayView src, CudaArrayView dst, TASK * task, const int & pivot, const Function & Cmp)
{
    static __shared__ int smallerStart, biggerStart;
    static __shared__ bool writePivot;

    int elemPerBlock = ceil( ((double)src.getSize()) / gridDim.x);
    int myBegin = blockIdx.x * elemPerBlock;
    int myEnd = TNL::min(src.getSize(), myBegin + elemPerBlock);

    int smaller = 0, bigger = 0;
    countElem(src, myBegin, myEnd, smaller, bigger, pivot);

    int smallerOffset = blockInclusivePrefixSum(smaller);
    int biggerOffset = blockInclusivePrefixSum(bigger);

    if (threadIdx.x == blockDim.x - 1) //last thread in block has sum of all values
    {
        smallerStart = atomicAdd(&(task->begin), smallerOffset);
        biggerStart = atomicAdd(&(task->end), -biggerOffset) - biggerOffset;
    }
    __syncthreads();

    int destSmaller = smallerStart + smallerOffset - smaller;
    int destBigger = biggerStart + biggerOffset - bigger;
    copyData(src, myBegin, myEnd, dst, destSmaller, destBigger, pivot);

    if (threadIdx.x == 0)
        writePivot = (atomicAdd(&(task->stillWorkingCnt), -1) == 1);
    __syncthreads();

    return writePivot;
}

template <typename Function>
__device__ void multiBlockQuickSort(CudaArrayView arr, CudaArrayView aux, TASK * task, const Function & Cmp, int depth)
{
    static __shared__ int pivot;

    if(threadIdx.x == 0)
        pivot = pickPivot(depth %2 == 0? arr: aux, Cmp);
    __syncthreads();
    
    bool isLast;
    if(depth %2 == 0)
        isLast = cudaPartition(arr, aux, task, pivot, Cmp);
    else
        isLast = cudaPartition(aux, arr, task, pivot, Cmp);

    if(!isLast)
        return;

    int leftEnd = task->begin, rightBegin = task->end;
    
    for (int i = leftEnd + threadIdx.x; i < rightBegin; i += blockDim.x)
        arr[i] = pivot;

    if(threadIdx.x != 0)
        return;
    
    int blocksLeft = 1, blocksRight = 1;
    calcBlocksNeeded(leftEnd - 0, arr.getSize() - rightBegin, blocksLeft, blocksRight);

    bool usedLeft = false;

    if(leftEnd > 0)
    {
        *task = TASK(0, leftEnd, blocksLeft);
        usedLeft = true;

        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cudaQuickSort<<<blocksLeft, blockDim.x, 0, s>>>(
                arr.getView(0, leftEnd),
                aux.getView(0, leftEnd),
                task,
                Cmp, depth+1);
        cudaStreamDestroy(s);
    }

    if((arr.getSize() - rightBegin)> 0)
    {
        TASK * newTaskRight = nullptr;

        if(usedLeft)
        {
            newTaskRight = (TASK * )malloc(sizeof(TASK));
            if(!newTaskRight)
            {
                printf("couldnt allocate memory for right task\n");
                return;
            }
            *newTaskRight = TASK(0, arr.getSize() - rightBegin, blocksRight);
        }
        else
        {
            usedLeft = true;
            *task = TASK(0, arr.getSize() - rightBegin, blocksRight);
        }

        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cudaQuickSort<<<blocksRight, blockDim.x, 0, s>>>(
                arr.getView(rightBegin, arr.getSize()),
                aux.getView(rightBegin, aux.getSize()), 
                newTaskRight? newTaskRight : task,
                Cmp, depth+1);
        cudaStreamDestroy(s);
    }

    if(!usedLeft)
        free(task);
}

//-------------------------------------------------------------------------
template <typename Function, int externMemSize>
__device__ void externSort(CudaArrayView src, CudaArrayView dst, const Function & Cmp)
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
    
    if(sizeL > sizeR)
    {
        if(sizeL > 0) //left from pivot are smaller elems
        {
            stackArrBegin[stackTop] = begin;
            stackArrEnd[stackTop] = pivotBegin;
            stackDepth[stackTop] = depth + 1;
            (stackTop)++;
        }
        
        if(sizeR > 0) //right from pivot until end are elem greater than pivot
        {
            assert(stackTop < stackSize && "Local quicksort stack overflow.");

            stackArrBegin[stackTop] = pivotEnd;
            stackArrEnd[stackTop] = end;
            stackDepth[stackTop] = depth + 1;
            (stackTop)++;
        }
    }
    else
    {
        if(sizeR > 0) //right from pivot until end are elem greater than pivot
        {
            stackArrBegin[stackTop] = pivotEnd;
            stackArrEnd[stackTop] = end;
            stackDepth[stackTop] = depth + 1;
            (stackTop)++;
        }

        if(sizeL > 0) //left from pivot are smaller elems
        {
            assert(stackTop < stackSize && "Local quicksort stack overflow.");

            stackArrBegin[stackTop] = begin;
            stackArrEnd[stackTop] = pivotBegin;
            stackDepth[stackTop] = depth + 1;
            (stackTop)++;
        }
    }
}

template <typename Function, int stackSize>
__device__ void singleBlockQuickSort(CudaArrayView arr, CudaArrayView aux, const Function & Cmp, int _depth)
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
            pivot = pickPivot(depth%2 == 0? 
                                    arr.getView(begin, end) :
                                    aux.getView(begin, end),
                                Cmp
                            );
        }
        __syncthreads();

        int size = end - begin;
        auto src = depth%2 == 0 ? arr.getView(begin, end) : aux.getView(begin, end);
        auto dst = depth%2 == 0 ? aux.getView(begin, end) : arr.getView(begin, end);

        if(size <= blockDim.x*2)
        {
            externSort<Function, 2048>(src, arr.getView(begin, end), Cmp);
            continue;
        }

        int smaller = 0, bigger = 0;
        countElem(src, 0, size, smaller, bigger, pivot);

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

        copyData(src, 0, size, dst, destSmaller, destBigger, pivot);
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

template <typename Function>
__global__ void cudaQuickSort(CudaArrayView arr, CudaArrayView aux, TASK * task, const Function & Cmp, int depth)
{
    if(gridDim.x > 1)
    {
        multiBlockQuickSort(arr, aux, task, Cmp, depth);
    }
    else
    {
        if(threadIdx.x == 0)
            free(task);
            
        singleBlockQuickSort<Function, 128>(arr, aux, Cmp, depth);
    }
}

//-----------------------------------------------------------

/**
 * call this kernel using 1 thread only
 * */
template <typename Function>
__global__ void cudaQuickSortEntry(CudaArrayView arr, CudaArrayView aux, const Function & Cmp, int blocks, int threadsPerBlock)
{
    TASK * task = (TASK *)malloc(sizeof(TASK));
    *task = TASK(0, arr.getSize(), blocks);
    if(!task)
    {
        printf("couldnt allocate memory for right task\n");
        return;
    }

    //task is freed by the block that wrote pivot
    cudaQuickSort<<<blocks, threadsPerBlock>>>(arr, aux, task, Cmp, 0);
}

//-----------------------------------------------------------

template<typename Function>
void quicksort(CudaArrayView arr, const Function & Cmp)
{
    TNL::Containers::Array<int, TNL::Devices::Cuda> aux(arr.getSize());
    
    const int threadsPerBlock = 512, maxBlocks = 1 << 15; //32k
    const int minElemPerBlock = threadsPerBlock*2;
    int sets = arr.getSize() / minElemPerBlock + (arr.getSize() % minElemPerBlock != 0);

    int blocks = min(sets, maxBlocks);
    cudaQuickSortEntry<<<1, 1>>>(arr, aux.getView(), Cmp, blocks, threadsPerBlock);
    cudaDeviceSynchronize();
}

void quicksort(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> arr)
{
    quicksort(arr, []__cuda_callable__(int a, int b){return a < b;});
}
