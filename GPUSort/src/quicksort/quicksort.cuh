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

__device__
void writeNewTask(int begin, int end, int depth, ArrayView<TASK, Devices::Cuda> newTasks, int *newTasksCnt,
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
                                      ArrayView<int, Devices::Cuda> taskMapping, int *tasksAmount,
                                      ArrayView<TASK, Devices::Cuda> newTasks, int *newTasksCnt,
                                      ArrayView<TASK, Devices::Cuda> secondPhaseTasks, int *secondPhaseTasksCnt)
{
    static __shared__ int pivot;
    TASK &myTask = tasks[taskMapping[blockIdx.x]];

    if (threadIdx.x == 0)
        pivot = pickPivot(myTask.depth %2 == 0? 
                            arr.getView(myTask.partitionBegin, myTask.partitionEnd ) :
                            aux.getView(myTask.partitionBegin, myTask.partitionEnd ),
                        Cmp
                    );
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
const int maxTasks = 1 << 10;
const int minElemPerBlock = threadsPerBlock * 2;

class QUICKSORT
{
    ArrayView<int, Devices::Cuda> arr;
    Array<int, Devices::Cuda> aux;

    Array<TASK, Devices::Cuda> cuda_tasks, cuda_newTasks, cuda_2ndPhaseTasks;
    Array<int, Devices::Cuda> cuda_newTasksAmount, cuda_2ndPhaseTasksAmount; //is in reality 1 integer

    int tasksAmount; //counter for Host == cuda_newTasksAmount
    int totalTask;   // cuda_newTasksAmount + cuda_2ndPhaseTasksAmount

    Array<int, Devices::Cuda> cuda_blockToTaskMapping;
    Array<int, Devices::Cuda> cuda_blockToTaskMapping_Cnt; //is in reality 1 integer

    int iteration = 0;

    //--------------------------------------------------------------------------------------
public:
    QUICKSORT(ArrayView<int, Devices::Cuda> _arr)
        : arr(_arr), aux(arr.getSize()),
          cuda_tasks(maxBlocks), cuda_newTasks(maxBlocks), cuda_2ndPhaseTasks(maxBlocks),
          cuda_newTasksAmount(1), cuda_2ndPhaseTasksAmount(1),
          cuda_blockToTaskMapping(maxBlocks), cuda_blockToTaskMapping_Cnt(1)
    {
        cuda_tasks.setElement(0, TASK(0, arr.getSize(), 0));
        totalTask = tasksAmount = 1;
        cuda_2ndPhaseTasksAmount = 0;
    }

    template <typename Function>
    void sort(const Function &cmp);
};

template <typename Function>
void QUICKSORT::sort(const Function &cmp)
{
    return;
}

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
