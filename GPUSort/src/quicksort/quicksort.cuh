#pragma once

#include <TNL/Containers/Array.h>
#include "../util/reduction.cuh"
#include "task.h"
#include "cudaPartition.cuh"
#include "quicksort_1Block.cuh"
#include "../bitonicSort/bitonicSort.h"
#include <iostream>

#define deb(x) std::cout << #x << " = " << x << std::endl;

using namespace TNL;
using namespace TNL::Containers;

//-----------------------------------------------------------

__device__ void writeNewTask(int begin, int end, int depth, ArrayView<TASK, Devices::Cuda> newTasks, int *newTasksCnt,
                             ArrayView<TASK, Devices::Cuda> secondPhaseTasks, int *secondPhaseTasksCnt)
{
    int size = end - begin;
    if (size == 0)
        return;
    if (size <= blockDim.x * 2)
    {
        int idx = atomicAdd(secondPhaseTasksCnt, 1);
        secondPhaseTasks[idx] = TASK(begin, end, depth + 1);
    }
    else
    {
        int idx = atomicAdd(newTasksCnt, 1);
        newTasks[idx] = TASK(begin, end, depth + 1);
    }
}

__device__ void writeNewTasks(int leftBegin, int leftEnd, int rightBegin, int rightEnd,
                              int depth,
                              ArrayView<TASK, Devices::Cuda> newTasks, int *newTasksCnt,
                              ArrayView<TASK, Devices::Cuda> secondPhaseTasks, int *secondPhaseTasksCnt)
{
    writeNewTask(leftBegin, leftEnd, depth, newTasks, newTasksCnt, secondPhaseTasks, secondPhaseTasksCnt);
    writeNewTask(rightBegin, rightEnd, depth, newTasks, newTasksCnt, secondPhaseTasks, secondPhaseTasksCnt);
}
//----------------------------------------------------

template <typename Function>
__global__ void cudaQuickSort1stPhase(ArrayView<int, Devices::Cuda> arr, ArrayView<int, Devices::Cuda> aux,
                                      const Function &Cmp, int elemPerBlock,
                                      ArrayView<TASK, Devices::Cuda> tasks,
                                      ArrayView<int, Devices::Cuda> taskMapping,
                                      ArrayView<TASK, Devices::Cuda> newTasks, int *newTasksCnt,
                                      ArrayView<TASK, Devices::Cuda> secondPhaseTasks, int *secondPhaseTasksCnt)
{
    static __shared__ int pivot;
    TASK &myTask = tasks[taskMapping[blockIdx.x]];

    if (threadIdx.x == 0)
    {
        pivot = pickPivot(myTask.depth % 2 == 0 ? arr.getView(myTask.partitionBegin, myTask.partitionEnd) : aux.getView(myTask.partitionBegin, myTask.partitionEnd),
                          Cmp);
    }
    __syncthreads();

    bool isLast;

    if (myTask.depth % 2 == 0)
    {
        isLast = cudaPartition(
            arr.getView(myTask.partitionBegin, myTask.partitionEnd),
            aux.getView(myTask.partitionBegin, myTask.partitionEnd),
            Cmp, pivot, elemPerBlock, myTask);
    }
    else
    {
        isLast = cudaPartition(
            aux.getView(myTask.partitionBegin, myTask.partitionEnd),
            arr.getView(myTask.partitionBegin, myTask.partitionEnd),
            Cmp, pivot, elemPerBlock, myTask);
    }

    if (!isLast)
        return;

    myTask = tasks[taskMapping[blockIdx.x]];

    int leftBegin = myTask.partitionBegin, leftEnd = myTask.partitionBegin + myTask.dstBegin;
    int rightBegin = myTask.partitionBegin + myTask.dstEnd, rightEnd = myTask.partitionEnd;

    for (int i = leftEnd + threadIdx.x; i < rightBegin; i += blockDim.x)
    {
        /*
        #ifdef DEBUG
        aux[i] = -1;
        #endif
        */
        aux[i] = -1;
        arr[i] = pivot;
    }

    if (threadIdx.x != 0)
        return;

    writeNewTasks(leftBegin, leftEnd, rightBegin, rightEnd,
                  myTask.depth,
                  newTasks, newTasksCnt,
                  secondPhaseTasks, secondPhaseTasksCnt);
}

//-----------------------------------------------------------

template <typename Function, int stackSize>
__global__ void cudaQuickSort2ndPhase(ArrayView<int, Devices::Cuda> arr, ArrayView<int, Devices::Cuda> aux,
                                      const Function &Cmp,
                                      ArrayView<TASK, Devices::Cuda> secondPhaseTasks)
{
    TASK &myTask = secondPhaseTasks[blockIdx.x];
    auto arrView = arr.getView(myTask.partitionBegin, myTask.partitionEnd);
    auto auxView = aux.getView(myTask.partitionBegin, myTask.partitionEnd);

    singleBlockQuickSort<Function, stackSize>(arrView, auxView, Cmp, myTask.depth);
}
//-----------------------------------------------------------

__global__ void cudaInitTask(ArrayView<TASK, Devices::Cuda> cuda_tasks,
                             int taskAmount, int elemPerBlock, int *firstAvailBlock,
                             ArrayView<int, Devices::Cuda> cuda_blockToTaskMapping)
{
    static __shared__ int avail;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int blocksNeeded = 0;

    if (i < taskAmount)
    {
        auto task = cuda_tasks[i];
        int size = task.partitionEnd - task.partitionBegin;
        blocksNeeded = size / elemPerBlock + (size % elemPerBlock != 0);
    }

    int blocksNeeded_total = blockInclusivePrefixSum(blocksNeeded);
    if (threadIdx.x == blockDim.x - 1)
        avail = atomicAdd(firstAvailBlock, blocksNeeded_total);
    __syncthreads();

    if (i < taskAmount)
    {
        int myFirstAvailBlock = avail + blocksNeeded_total - blocksNeeded;

        cuda_tasks[i].initTask(myFirstAvailBlock, blocksNeeded);

        for (int set = 0; set < blocksNeeded; set++)
            cuda_blockToTaskMapping[myFirstAvailBlock++] = i;
    }
}

//-----------------------------------------------------------
//-----------------------------------------------------------
const int threadsPerBlock = 512, maxBlocks = 1 << 15; //32k
const int g_maxTasks = 1 << 14;
const int minElemPerBlock = threadsPerBlock;

class QUICKSORT
{
    ArrayView<int, Devices::Cuda> arr;
    Array<int, Devices::Cuda> aux;
    int maxTasks;
    Array<TASK, Devices::Cuda> cuda_tasks, cuda_newTasks, cuda_2ndPhaseTasks;
    Array<int, Devices::Cuda> cudaCounters;

    ArrayView<int, Devices::Cuda> cuda_newTasksAmount, cuda_2ndPhaseTasksAmount; //is in reality 1 integer

    int tasksAmount; //counter for Host == cuda_newTasksAmount
    int totalTask;   // cuda_newTasksAmount + cuda_2ndPhaseTasksAmount

    Array<int, Devices::Cuda> cuda_blockToTaskMapping;
    ArrayView<int, Devices::Cuda> cuda_blockToTaskMapping_Cnt; //is in reality 1 integer

    int iteration = 0;

    //--------------------------------------------------------------------------------------
public:
    QUICKSORT(ArrayView<int, Devices::Cuda> _arr)
        : arr(_arr), aux(arr.getSize()),
          maxTasks(min(arr.getSize(), g_maxTasks)),
          cuda_tasks(maxTasks), cuda_newTasks(maxTasks), cuda_2ndPhaseTasks(maxTasks),
          cudaCounters(3),
          cuda_newTasksAmount(cudaCounters.getView(0, 1)),
          cuda_2ndPhaseTasksAmount(cudaCounters.getView(1, 2)),
          cuda_blockToTaskMapping(maxTasks),
          cuda_blockToTaskMapping_Cnt(cudaCounters.getView(2, 3))
    {
        cuda_tasks.setElement(0, TASK(0, arr.getSize(), 0));
        totalTask = tasksAmount = 1;
        cuda_2ndPhaseTasksAmount = 0;
        iteration = 0;
    }

    template <typename Function>
    void sort(const Function &Cmp);

    int getSetsNeeded() const;
    int getElemPerBlock() const;

    /**
     * returns the amount of blocks needed
     * */
    int initTasks(int elemPerBlock);

    void processNewTasks();
};

template <typename Function>
void QUICKSORT::sort(const Function &Cmp)
{
    
    while (tasksAmount > 0 && tasksAmount*2 < maxTasks)
    {
        int elemPerBlock = getElemPerBlock();
        int blocksCnt = initTasks(elemPerBlock);
        if (iteration % 2 == 0)
        {
            cudaQuickSort1stPhase<Function>
                <<<blocksCnt, threadsPerBlock>>>(
                    arr, aux, Cmp, elemPerBlock,
                    cuda_tasks,
                    cuda_blockToTaskMapping,
                    cuda_newTasks,
                    cuda_newTasksAmount.getData(),
                    cuda_2ndPhaseTasks, cuda_2ndPhaseTasksAmount.getData());
        }
        else
        {
            cudaQuickSort1stPhase<<<blocksCnt, threadsPerBlock>>>(
                arr, aux, Cmp, elemPerBlock,
                cuda_newTasks,
                cuda_blockToTaskMapping,
                cuda_tasks,
                cuda_newTasksAmount.getData(),
                cuda_2ndPhaseTasks, cuda_2ndPhaseTasksAmount.getData());
        }
        processNewTasks();
        iteration++;
    }
    
    if(tasksAmount > 0)
    {
        cudaQuickSort2ndPhase<Function, 128>
            <<<tasksAmount, threadsPerBlock>>>(arr, aux, Cmp,
            iteration % 2 == 0? cuda_newTasks : cuda_tasks
        );
    }

    if(totalTask - tasksAmount > 0)
    {
        cudaQuickSort2ndPhase<Function, 128>
            <<<totalTask - tasksAmount, threadsPerBlock>>>(arr, aux, Cmp, cuda_2ndPhaseTasks);
    }
        
    cudaDeviceSynchronize();
    return;
}

int QUICKSORT::getSetsNeeded() const
{
    auto view = iteration % 2 == 0 ? cuda_tasks.getConstView() : cuda_newTasks.getConstView();
    auto fetch = [=] __cuda_callable__(int i) {
        auto &task = view[i];
        int size = task.partitionEnd - task.partitionBegin;
        return size / minElemPerBlock + (size % minElemPerBlock != 0);
    };
    auto reduction = [] __cuda_callable__(int a, int b) { return a + b; };
    return Algorithms::Reduction<Devices::Cuda>::reduce(0, tasksAmount, reduction, fetch, 0);
}

int QUICKSORT::getElemPerBlock() const
{
    int setsNeeded = getSetsNeeded();

    if (setsNeeded <= maxBlocks)
        return minElemPerBlock;

    int setsPerBlock = ceil(1. * setsNeeded / maxBlocks);
    return setsPerBlock * minElemPerBlock;
}

int QUICKSORT::initTasks(int elemPerBlock)
{
    int threads = min(tasksAmount, threadsPerBlock);
    int blocks = tasksAmount / threads + (tasksAmount % threads != 0);
    cuda_blockToTaskMapping_Cnt = 0;

    if (iteration % 2 == 0)
    {
        cudaInitTask<<<blocks, threads>>>(
            cuda_tasks, tasksAmount, elemPerBlock,
            cuda_blockToTaskMapping_Cnt.getData(),
            cuda_blockToTaskMapping);
    }
    else
    {
        cudaInitTask<<<blocks, threads>>>(
            cuda_newTasks, tasksAmount, elemPerBlock,
            cuda_blockToTaskMapping_Cnt.getData(),
            cuda_blockToTaskMapping);
    }

    cuda_newTasksAmount.setElement(0, 0);
    return cuda_blockToTaskMapping_Cnt.getElement(0);
}

void QUICKSORT::processNewTasks()
{
    tasksAmount = cuda_newTasksAmount.getElement(0);
    totalTask = tasksAmount + cuda_2ndPhaseTasksAmount.getElement(0);
}

//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------

template <typename Function>
void quicksort(ArrayView<int, Devices::Cuda> arr, const Function &Cmp)
{
    QUICKSORT sorter(arr);
    sorter.sort(Cmp);
}

void quicksort(ArrayView<int, Devices::Cuda> arr)
{
    quicksort(arr, [] __cuda_callable__(int a, int b) { return a < b; });
}
