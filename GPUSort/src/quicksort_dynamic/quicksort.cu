#include "quicksort.cuh"

#include <TNL/Containers/Array.h>
#include "../util/reduction.cuh"
#include "task.h"
#include "../bitonicSort/bitonicSort.h"
#include "helper.cuh"
#include <iostream>
#include <cmath>
#include <cassert>

#define deb(x) std::cout << #x << " = " << x << std::endl;

using namespace TNL;
using namespace TNL::Containers;

template <typename Function>
__global__ void cudaPartition(ArrayView<int, Devices::Cuda> src, ArrayView<int, Devices::Cuda> dst, int pivot, TASK *task, const Function &Cmp)
{
    static __shared__ int smallerStart, biggerStart;

    int elemPerBlock = ceil(((double)src.getSize()) / gridDim.x);
    int myBegin = blockIdx.x * elemPerBlock;
    int myEnd = TNL::min(src.getSize(), myBegin + elemPerBlock);

    int smaller = 0, bigger = 0;
    countElem(src.getView(myBegin, myEnd), smaller, bigger, pivot);

    int smallerInclusiveSum = blockInclusivePrefixSum(smaller);
    int biggerInclusiveSum = blockInclusivePrefixSum(bigger);

    if (threadIdx.x == blockDim.x - 1) //last thread in block has sum of all values
    {
        smallerStart = atomicAdd(&(task->begin), smallerInclusiveSum);
        biggerStart = atomicAdd(&(task->end), -biggerInclusiveSum) - biggerInclusiveSum;
    }
    __syncthreads();

    int destSmaller = smallerStart + (smallerInclusiveSum - smaller);
    int destBigger = biggerStart + (biggerInclusiveSum - bigger);
    copyData(src.getView(myBegin, myEnd), dst, destSmaller, destBigger, pivot);
}

template <typename Function>
__device__ void multiBlockQuickSort(ArrayView<int, Devices::Cuda> arr, ArrayView<int, Devices::Cuda> aux, const Function &Cmp, int depth, int availblocks)
{
    static __shared__ int pivot;
    static __shared__ int leftEnd, rightBegin;

    if (threadIdx.x == 0)
    {
        pivot = pickPivot(depth % 2 == 0 ? arr : aux, Cmp);

        TASK *task = (TASK *)malloc(sizeof(TASK));
        *task = TASK(0, arr.getSize());

        if (depth % 2 == 0)
            cudaPartition<<<availblocks, 512>>>(arr, aux, pivot, task, Cmp);
        else
            cudaPartition<<<availblocks, 512>>>(aux, arr, pivot, task, Cmp);
        cudaDeviceSynchronize();

        leftEnd = task->begin, rightBegin = task->end;
        free(task);
    }
    __syncthreads();

    for (int i = leftEnd + threadIdx.x; i < rightBegin; i += blockDim.x)
        arr[i] = pivot;

    if (threadIdx.x == 0)
    {
        int blocksLeft = 0, blocksRight = 0;
        calcBlocksNeeded(availblocks, leftEnd - 0, arr.getSize() - rightBegin, blocksLeft, blocksRight);

        if(leftEnd > 0)
        {
            cudaStream_t s;
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);

            cudaQuickSort<<<1, blockDim.x, 0, s>>>(arr.getView(0, leftEnd), aux.getView(0, leftEnd), Cmp, blocksLeft, depth + 1);
            
            cudaStreamDestroy(s);
        }
        if(arr.getSize() - rightBegin > 0)
        {
            cudaStream_t s;
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);

            cudaQuickSort<<<1, blockDim.x, 0, s>>>(arr.getView(rightBegin, arr.getSize()), aux.getView(rightBegin, aux.getSize()), Cmp, blocksRight, depth + 1);
            cudaStreamDestroy(s);
        }
    }
}

//-------------------------------------------------------------------------

template <typename Function, int externMemSize>
__device__ void externSort(ArrayView<int, Devices::Cuda> src, ArrayView<int, Devices::Cuda> dst, const Function &Cmp)
{
    static __shared__ int sharedMem[externMemSize];
    bitonicSort_Block(src, dst, sharedMem, Cmp);
}

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
            (stackTop)++;
        }

        if (sizeR > 0) //right from pivot until end are elem greater than pivot
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
        if (sizeR > 0) //right from pivot until end are elem greater than pivot
        {
            stackArrBegin[stackTop] = pivotEnd;
            stackArrEnd[stackTop] = end;
            stackDepth[stackTop] = depth + 1;
            (stackTop)++;
        }

        if (sizeL > 0) //left from pivot are smaller elems
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
__device__ void singleBlockQuickSort(ArrayView<int, Devices::Cuda> arr, ArrayView<int, Devices::Cuda> aux, const Function &Cmp, int _depth)
{
    static __shared__ int stackTop;
    static __shared__ int stackArrBegin[stackSize], stackArrEnd[stackSize], stackDepth[stackSize];
    static __shared__ int begin, end, depth, pivotBegin, pivotEnd;
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

    while (stackTop > 0)
    {
        if (threadIdx.x == 0)
        {
            begin = stackArrBegin[stackTop - 1];
            end = stackArrEnd[stackTop - 1];
            depth = stackDepth[stackTop - 1];
            stackTop--;
            pivot = pickPivot(depth % 2 == 0 ? arr.getView(begin, end) : aux.getView(begin, end),
                              Cmp);
        }
        __syncthreads();

        int size = end - begin;
        auto src = depth % 2 == 0 ? arr.getView(begin, end) : aux.getView(begin, end);
        auto dst = depth % 2 == 0 ? aux.getView(begin, end) : arr.getView(begin, end);

        if (size <= blockDim.x * 2)
        {
            externSort<Function, 1024>(src, arr.getView(begin, end), Cmp);
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
        int destBigger = pivotEnd + (biggerOffset - bigger);

        copyData(src, dst, destSmaller, destBigger, pivot);
        __syncthreads();

        for (int i = pivotBegin + threadIdx.x; i < pivotEnd; i += blockDim.x)
            src[i] = dst[i] = pivot;

        if (threadIdx.x == 0)
        {
            stackPush<stackSize>(stackArrBegin, stackArrEnd, stackDepth, stackTop,
                                 begin, begin + pivotBegin,
                                 begin + pivotEnd, end,
                                 depth);
        }
        __syncthreads();
    } //ends while loop
}

//-------------------------------------------------------------------------

template <typename Function>
__global__ void cudaQuickSort(ArrayView<int, Devices::Cuda> arr, ArrayView<int, Devices::Cuda> aux,
                              const Function &Cmp, int availBlocks, int depth)
{
    if (availBlocks == 0 || arr.getSize() <= blockDim.x * 2 || depth >= 4) //todo: determine max depth
        singleBlockQuickSort<Function, 128>(arr, aux, Cmp, depth);
    else
        multiBlockQuickSort(arr, aux, Cmp, depth, availBlocks);
}

//-----------------------------------------------------------

template <typename Function>
void quicksort(ArrayView<int, Devices::Cuda> arr, const Function &Cmp)
{
    TNL::Containers::Array<int, TNL::Devices::Cuda> aux(arr.getSize());

    const int threadsPerBlock = 512, maxBlocks = 1 << 15; //32k
    const int minElemPerBlock = threadsPerBlock * 2;
    int sets = arr.getSize() / minElemPerBlock + (arr.getSize() % minElemPerBlock != 0);

    int blocks = min(sets, maxBlocks);
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 10);
    cudaQuickSort<<<1, threadsPerBlock>>>(arr, aux.getView(), Cmp, blocks, 0);
    cudaDeviceSynchronize();
}

void quicksort(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> arr)
{
    quicksort(arr, [] __cuda_callable__(int a, int b) { return a < b; });
}
