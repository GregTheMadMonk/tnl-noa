#pragma once

#include <TNL/Containers/Array.h>
#include "reduction.cuh"
#include "task.h"
#include <iostream>

#define deb(x) std::cout << #x << " = " << x << std::endl;

using CudaArrayView = TNL::Containers::ArrayView<int, TNL::Devices::Cuda>;

__device__ void cmpElem(CudaArrayView arr, int myBegin, int myEnd,
                        int &smaller, int &bigger,
                        volatile int pivot)
{
    for (int i = myBegin + threadIdx.x; i < myEnd; i += blockDim.x)
    {
        int data = arr[i];
        if (data < pivot)
            smaller++;
        else if (data > pivot)
            bigger++;
    }
}

__device__ void copyData(CudaArrayView arr, int myBegin, int myEnd,
                         CudaArrayView aux, int smallerStart, int biggerStart,
                         volatile int pivot)
{
    for (int i = myBegin + threadIdx.x; i < myEnd; i += blockDim.x)
    {
        int data = arr[i];
        if (data < pivot)
            aux[smallerStart++] = data;
        else if (data > pivot)
            aux[biggerStart++] = data;
    }
}

__global__ void cudaPartition(CudaArrayView arr, CudaArrayView aux, int elemPerBlock,
                              TNL::Containers::ArrayView<TASK, TNL::Devices::Cuda> cuda_tasks,
                              TNL::Containers::ArrayView<int, TNL::Devices::Cuda> cuda_blockToTaskMapping,
                              TNL::Containers::ArrayView<TASK, TNL::Devices::Cuda> cuda_newTasks,
                              int *newTasksCnt)
{
    static __shared__ int smallerStart, biggerStart;
    static __shared__ int pivot;
    static __shared__ int myTaskIdx;
    static __shared__ TASK myTask;
    static __shared__ bool writePivot;

    if (threadIdx.x == 0)
    {
        myTaskIdx = cuda_blockToTaskMapping[blockIdx.x];
        myTask = cuda_tasks[myTaskIdx];
        pivot = myTask.pivot;
        writePivot = false;
    }
    __syncthreads();

    //only works if consecutive blocks work on the same task
    const int myBegin = myTask.arrBegin + elemPerBlock * (blockIdx.x - myTask.firstBlock);
    const int myEnd = TNL::min(myTask.arrEnd, myBegin + elemPerBlock);

    int smaller = 0, bigger = 0;
    cmpElem(arr, myBegin, myEnd, smaller, bigger, pivot);

    int smallerOffset = blockInclusivePrefixSum(smaller);
    int biggerOffset = blockInclusivePrefixSum(bigger);

    if (threadIdx.x == blockDim.x - 1) //last thread in block has sum of all values
    {
        smallerStart = atomicAdd(&(cuda_tasks[myTaskIdx].auxBeginIdx), smallerOffset);
        biggerStart = atomicAdd(&(cuda_tasks[myTaskIdx].auxEndIdx), -biggerOffset) - biggerOffset;
    }
    __syncthreads();

    int destSmaller = smallerStart + smallerOffset - smaller;
    int destBigger = biggerStart + biggerOffset - bigger;
    copyData(arr, myBegin, myEnd, aux, destSmaller, destBigger, pivot);

    if (threadIdx.x == 0 && atomicAdd(&(cuda_tasks[myTaskIdx].blockCount), -1) == 1)
    {
        writePivot = true;
        myTask = cuda_tasks[myTaskIdx];
    }
    __syncthreads();

    if (!writePivot)
        return;

    for (int i = myTask.auxBeginIdx + threadIdx.x; i < myTask.auxEndIdx; i += blockDim.x)
    {
        arr[i] = pivot;
        aux[i] = pivot;
    }

    //only works if aux array is as big as input array
    if (threadIdx.x == 0)
    {
        if (myTask.auxBeginIdx - myTask.arrBegin > 0)
        {
            int newTaskIdx = atomicAdd(newTasksCnt, 1);
            cuda_newTasks[newTaskIdx] = TASK(
                myTask.arrBegin, myTask.auxBeginIdx,
                myTask.arrBegin, myTask.auxBeginIdx,
                aux[myTask.auxBeginIdx - 1]);
        }

        if (myTask.arrEnd - myTask.auxEndIdx > 0)
        {
            int newTaskIdx = atomicAdd(newTasksCnt, 1);
            cuda_newTasks[newTaskIdx] = TASK(
                myTask.auxEndIdx, myTask.arrEnd,
                myTask.auxEndIdx, myTask.arrEnd,
                aux[myTask.arrEnd - 1]);
        }
    }
}

__global__ void cudaInitTask(TNL::Containers::ArrayView<TASK, TNL::Devices::Cuda> cuda_tasks,
                        int taskAmount, int elemPerBlock, int *firstAvailBlock,
                        TNL::Containers::ArrayView<int, TNL::Devices::Cuda> cuda_blockToTaskMapping)
{
    static __shared__ int avail;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int blocksNeeded = 0;

    if(i < taskAmount)
    {
        auto task = cuda_tasks[i];
        int size = task.arrEnd - task.arrBegin;
        blocksNeeded = size / elemPerBlock + (size % elemPerBlock != 0);
    }

    int blocksNeeded_total = blockInclusivePrefixSum(blocksNeeded);
    if(threadIdx.x == blockDim.x - 1)
        avail = atomicAdd(firstAvailBlock, blocksNeeded_total);
    __syncthreads();
        
    if(i < taskAmount)
    {
        int myFirstAvailBlock = avail + blocksNeeded_total - blocksNeeded;

        cuda_tasks[i].firstBlock = myFirstAvailBlock;
        cuda_tasks[i].blockCount = blocksNeeded;

        for (int set = 0; set < blocksNeeded; set++)
            cuda_blockToTaskMapping[myFirstAvailBlock++] = i;
    }
}

//-----------------------------------------------------------
const int threadsPerBlock = 512, maxBlocks = 1 << 14; //16k
const int maxTasks = 1 << 20;

class QUICKSORT
{
    CudaArrayView arr;
    TNL::Containers::Array<int, TNL::Devices::Cuda> aux;

    TNL::Containers::Array<TASK, TNL::Devices::Cuda> cuda_tasks;
    TNL::Containers::Array<TASK, TNL::Devices::Cuda> cuda_newTasks;

    TNL::Containers::Array<int, TNL::Devices::Cuda> cuda_newTasksAmount; //is in reality 1 integer
    int tasksAmount;

    TNL::Containers::Array<int, TNL::Devices::Cuda> cuda_blockToTaskMapping;
    TNL::Containers::Array<int, TNL::Devices::Cuda> cuda_blockToTaskMapping_Cnt; //is in reality 1 integer
    int iteration = 0;
    //--------------------------------------------------------------------------------------
public:
    QUICKSORT(CudaArrayView _arr)
        : arr(_arr), aux(arr.getSize()),
          cuda_tasks(maxTasks), cuda_newTasks(maxTasks), cuda_newTasksAmount(1),
          cuda_blockToTaskMapping(maxBlocks), cuda_blockToTaskMapping_Cnt(1)
    {
        int pivot = arr.getElement(arr.getSize() - 1);
        cuda_tasks.setElement(0, TASK(0, arr.getSize(), 0, arr.getSize(), pivot));
        tasksAmount = 1;
    }


    void sort()
    {
        while (tasksAmount > 0)
        {
            int elemPerBlock = getBlockSize();
            int blocksCnt = initTasks(elemPerBlock);

            if(iteration%2 == 0)
            {
                cudaPartition<<<blocksCnt, threadsPerBlock>>>(arr, aux.getView(), elemPerBlock,
                    cuda_tasks.getView(), cuda_blockToTaskMapping.getView(),
                    cuda_newTasks.getView(), cuda_newTasksAmount.getData());
            }
            else
            {
                cudaPartition<<<blocksCnt, threadsPerBlock>>>(aux.getView(), arr, elemPerBlock,
                    cuda_newTasks.getView(), cuda_blockToTaskMapping.getView(),
                    cuda_tasks.getView(), cuda_newTasksAmount.getData());
            }
            
            tasksAmount = processNewTasks();

            iteration++;
        }

        if(iteration%2)
        {
            TNL::Algorithms::MultiDeviceMemoryOperations<TNL::Devices::Cuda, TNL::Devices::Cuda>::
            copy(arr.getData(), aux.getData(), aux.getSize());
        }

        cudaDeviceSynchronize();
    }

    int getSetsNeeded() const
    {
        auto view = iteration%2 == 0? cuda_tasks.getConstView() : cuda_newTasks.getConstView();
        auto fetch = [=] __cuda_callable__ (int i) {
            auto & task = view[i];
            int size = task.arrEnd - task.arrBegin;
            return size / threadsPerBlock + (size % threadsPerBlock != 0); 
        };
        auto reduction = [] __cuda_callable__(int a, int b) {return a + b;};
        return TNL::Algorithms::Reduction<TNL::Devices::Cuda>::reduce(0, tasksAmount, reduction, fetch, 0);
    }

    int getBlockSize() const
    {
        int setsNeeded = getSetsNeeded();

        if (setsNeeded <= maxBlocks)
            return threadsPerBlock;

        int setsPerBlock = setsNeeded / maxBlocks + 1; //+1 to spread out task of the last block
        return setsPerBlock * threadsPerBlock;
    }

    int initTasks(int elemPerBlock)
    {
        int threads = min(tasksAmount, 512);
        int blocks = tasksAmount / threads + (tasksAmount % threads != 0);
        cuda_blockToTaskMapping_Cnt = 0;

        if(iteration%2 == 0)
        {
            cudaInitTask<<<blocks,threads>>>(
                cuda_tasks.getView(), tasksAmount, elemPerBlock,
                cuda_blockToTaskMapping_Cnt.getData(), 
                cuda_blockToTaskMapping.getView());
        }
        else
        {
            cudaInitTask<<<blocks,threads>>>(
                cuda_newTasks.getView(), tasksAmount, elemPerBlock,
                cuda_blockToTaskMapping_Cnt.getData(), 
                cuda_blockToTaskMapping.getView());
        }

        cuda_newTasksAmount = 0;
        return cuda_blockToTaskMapping_Cnt.getElement(0);
    }

    int processNewTasks()
    {
        /*
        TNL::Algorithms::MultiDeviceMemoryOperations<TNL::Devices::Cuda, TNL::Devices::Cuda>::
            copy(arr.getData(), aux.getData(), aux.getSize());
        */
        tasksAmount = cuda_newTasksAmount.getElement(0);

        /*
        TNL::Algorithms::MultiDeviceMemoryOperations<TNL::Devices::Host, TNL::Devices::Cuda>::
            copy(cuda_tasks.getData(), cuda_newTasks.getData(), tasksAmount);
        */
        return tasksAmount;
    }

};

//-----------------------------------------------------------

void quicksort(CudaArrayView arr)
{
    QUICKSORT sorter(arr);
    sorter.sort();
    return;
}
