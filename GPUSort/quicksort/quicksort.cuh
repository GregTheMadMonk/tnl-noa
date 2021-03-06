#pragma once

#include <TNL/Containers/Array.h>
#include "reduction.cuh"
#include "task.h"
#include <utility>

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

//-----------------------------------------------------------
class QUICKSORT
{
    static const int threadsPerBlock = 512, maxBlocks = 1 << 14; //16k
    const int maxTasks = 1<<20;

    CudaArrayView arr;
    int begin, end;
    int size;
    TNL::Containers::Array<int, TNL::Devices::Cuda> aux;

    TNL::Containers::Array<TASK, TNL::Devices::Host> host_tasks;
    TNL::Containers::Array<TASK, TNL::Devices::Cuda> cuda_tasks;
    TNL::Containers::Array<TASK, TNL::Devices::Cuda> newTasks;
    int tasksAmount;

    TNL::Containers::Array<int, TNL::Devices::Host> host_blockToTaskMapping;
    TNL::Containers::Array<int, TNL::Devices::Cuda> cuda_blockToTaskMapping;

    //--------------------------------------------------------------------------------------

    int getSetsNeeded()
    {
        auto view = host_tasks.getView();
        auto fetch = [=] __cuda_callable__ (int i) {
            auto task = view.getElement(i);
            int size = task.arrEnd - task.arrBegin;
            return size / threadsPerBlock + (size % threadsPerBlock != 0); 
        };
        auto reduction = [] __cuda_callable__(int a, int b) {return a + b;};
        return TNL::Algorithms::Reduction<TNL::Devices::Host>::reduce(0, tasksAmount, reduction, fetch, 0);
    }
    
    std::pair<int, int> calcConfig()
    {
        int setsNeeded = getSetsNeeded();

        if(setsNeeded <= maxBlocks)
            return {setsNeeded, threadsPerBlock};

        int setsPerBlock = setsNeeded / maxBlocks + 1; //+1 to spread out task of the last block
        int elemPerBlock = setsPerBlock * threadsPerBlock;
        int blocks = size / elemPerBlock + (size % elemPerBlock != 0);
        return {blocks, elemPerBlock};
    }

    int initTasks(std::pair<int, int> blocks_elemPerBlock)
    {
        int elemPerBlock = blocks_elemPerBlock.second;
        auto host_tasksView = host_tasks.getView();
        int blockToTaskMapping_Cnt = 0;

        for(int i = 0; i < tasksAmount; ++i)
        {
            TASK & task = host_tasks[i];
            int size = task.arrEnd - task.arrBegin;
            int blocksNeeded = size / elemPerBlock + (size % elemPerBlock != 0);

            task.firstBlock = blockToTaskMapping_Cnt;
            task.blockCount = blocksNeeded;

            for(int set = 0; set < blocksNeeded; set++)
                host_blockToTaskMapping[blockToTaskMapping_Cnt++] = i;
        }

        TNL::Algorithms::MultiDeviceMemoryOperations<TNL::Devices::Cuda, TNL::Devices::Host >::
        copy(cuda_tasks.getData(), host_tasks.getData(), host_tasks.getSize());

        TNL::Algorithms::MultiDeviceMemoryOperations<TNL::Devices::Cuda, TNL::Devices::Host >::
        copy(cuda_blockToTaskMapping.getData(), host_blockToTaskMapping.getData(), host_blockToTaskMapping.getSize());

        if(blockToTaskMapping_Cnt != blocks_elemPerBlock.first)
        {
            std::cerr << "blockToTaskMapping_Cnt != blocks_elemPerBlock" << std::endl;
            class INVALID_CONFIG{};
            throw INVALID_CONFIG();
        }

        return blockToTaskMapping_Cnt;
    }

public:
    QUICKSORT(CudaArrayView arr, int begin, int end)
        : arr(arr.getView()), begin(begin), end(end),
        size(end - begin), aux(size),
        host_tasks(maxTasks), cuda_tasks(maxTasks), newTasks(maxTasks),
        host_blockToTaskMapping(maxBlocks), cuda_blockToTaskMapping(maxBlocks)
    {
        int pivotIdx = end - 1;
        host_tasks[0] = TASK(begin, end, 0, size, pivotIdx);
        tasksAmount = 1;
    }

    void sort()
    {
        while(tasksAmount > 0)
        {
            std::pair<int, int> blocks_elemPerBlock = calcConfig();
            int blocksCnt = initTasks(blocks_elemPerBlock);
            /*
            partition(arr, aux.getView(),
                cuda_tasks.getView(), cuda_blockToTaskMapping.getView()
                newTasks.getView());
            */
           processTasks();
        }

        //2nd phase to finish
    }

};

//-----------------------------------------------------------

void quicksort(CudaArrayView arr)
{
    quicksort(arr, 0, arr.getSize());
}
