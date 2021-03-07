#pragma once

#include <TNL/Containers/Array.h>
#include "reduction.cuh"
#include "task.h"
#include <utility>

#define deb(x) std::cout << #x << " = " << x << std::endl;

using CudaArrayView = TNL::Containers::ArrayView<int, TNL::Devices::Cuda>;

__device__ void cmpElem(CudaArrayView arr, int myBegin, int myEnd,
                        int &smaller, int &bigger,
                        int pivot)
{
    for (int i = myBegin + threadIdx.x; i < myEnd; i += blockDim.x)
    {
        int data = arr[i];
        if (data < pivot)
            smaller++;
        else if(data > pivot)
            bigger++;
    }
}

__device__ void copyData(CudaArrayView arr, int myBegin, int myEnd,
                         CudaArrayView aux, int smallerStart, int biggerStart,
                         int pivot)
{
    for (int i = myBegin + threadIdx.x; i < myEnd; i += blockDim.x)
    {
        int data = arr[i];
        if (data < pivot)
            aux[smallerStart++] = data;
        else if(data > pivot)
            aux[biggerStart++] = data;
    }
}

__global__
void cudaPartition(CudaArrayView arr, CudaArrayView aux, int elemPerBlock,
                TNL::Containers::ArrayView<TASK, TNL::Devices::Cuda> cuda_tasks,
                TNL::Containers::ArrayView<int, TNL::Devices::Cuda> cuda_blockToTaskMapping,
                TNL::Containers::ArrayView<TASK, TNL::Devices::Cuda> cuda_newTasks,
                int * newTasksCnt)
{
    static __shared__ int smallerStart, biggerStart;
    static __shared__ int pivot;
    static __shared__ int myTaskIdx;
    static __shared__ TASK myTask;
    static __shared__ bool writePivot;

    if(threadIdx.x == 0)
    {
        myTaskIdx = cuda_blockToTaskMapping[blockIdx.x];
        myTask = cuda_tasks[myTaskIdx];
        pivot = arr[myTask.pivotPos];
        writePivot = false;
    }
    __syncthreads();

    const int myBegin = myTask.arrBegin + elemPerBlock * (blockIdx.x - myTask.firstBlock);
    const int myEnd = TNL::min(myTask.arrEnd, myBegin + elemPerBlock);

    int smaller = 0, bigger = 0;
    cmpElem(arr, myBegin, myEnd, pivot, smaller, bigger);

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

    if(threadIdx.x == 0 && atomicAdd(&(cuda_tasks[myTaskIdx].blockCount), -1) == 1)
    {
        writePivot = true;
        myTask = cuda_tasks[myTaskIdx];
    }
    __syncthreads();

    if(!writePivot)
        return;
    
    for(int i = myTask.auxBeginIdx + threadIdx.x; i < myTask.auxEndIdx; i+= blockDim.x)
        aux[i] = pivot;

    //only works if aux array is as big as input array
    if(threadIdx.x == 0)
    {
        if(myTask.auxBeginIdx - myTask.arrBegin > 1)
        {
            int newTaskIdx = atomicAdd(newTasksCnt, 1);
            cuda_newTasks[newTaskIdx] = TASK(
                    myTask.arrBegin, myTask.auxBeginIdx, 
                    myTask.arrBegin, myTask.auxBeginIdx,
                    myTask.auxBeginIdx - 1);
        }

        if(myTask.arrEnd - myTask.auxEndIdx > 1)
        {
            int newTaskIdx = atomicAdd(newTasksCnt, 1);
            cuda_newTasks[newTaskIdx] = TASK(
                    myTask.auxEndIdx, myTask.arrEnd, 
                    myTask.auxEndIdx, myTask.arrEnd, 
                    myTask.arrEnd - 1);
        }
    }

}

//-----------------------------------------------------------
class QUICKSORT
{
    static const int threadsPerBlock = 512, maxBlocks = 1 << 14; //16k
    const int maxTasks = 1<<20;

    CudaArrayView arr;
    TNL::Containers::Array<int, TNL::Devices::Cuda> aux;

    TNL::Containers::Array<TASK, TNL::Devices::Host> host_tasks;
    TNL::Containers::Array<TASK, TNL::Devices::Cuda> cuda_tasks;
    TNL::Containers::Array<TASK, TNL::Devices::Cuda> newTasks;
    int tasksAmount;

    TNL::Containers::Array<int, TNL::Devices::Host> host_blockToTaskMapping;
    TNL::Containers::Array<int, TNL::Devices::Cuda> cuda_blockToTaskMapping;

    //--------------------------------------------------------------------------------------
public:

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
    
    int getBlockSize()
    {
        int setsNeeded = getSetsNeeded();

        if(setsNeeded <= maxBlocks)
            return threadsPerBlock;

        int setsPerBlock = setsNeeded / maxBlocks + 1; //+1 to spread out task of the last block
        return setsPerBlock * threadsPerBlock;
    }

    int initTasks(int elemPerBlock)
    {
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

        return blockToTaskMapping_Cnt;
    }

    QUICKSORT(CudaArrayView arr)
        : arr(arr.getView()), aux(arr.getSize()),
        host_tasks(maxTasks), cuda_tasks(maxTasks), newTasks(maxTasks),
        host_blockToTaskMapping(maxBlocks), cuda_blockToTaskMapping(maxBlocks)
    {
        int pivotIdx = arr.getSize() - 1;
        host_tasks[0] = TASK(0, arr.getSize(), 0, arr.getSize(), pivotIdx);
        tasksAmount = 1;
    }

    void sort()
    {
        while(tasksAmount > 0)
        {
            int elemPerBlock = getBlockSize();
            int blocksCnt = initTasks(elemPerBlock);

            /*
            partition(arr, aux.getView(),
                cuda_tasks.getView(), cuda_blockToTaskMapping.getView()
                newTasks.getView());
            */
           
            //processTasks();
        }

        //2nd phase to finish
    }

};

//-----------------------------------------------------------

void quicksort(CudaArrayView arr)
{
    //quicksort(arr, 0, arr.getSize());
    return;
}
