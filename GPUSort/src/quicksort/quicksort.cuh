#pragma once

#include <TNL/Containers/Array.h>
#include "../util/reduction.cuh"
#include "task.h"
#include "cudaPartition.cuh"
#include "../bitonicSort/bitonicSort.h"
#include <iostream>

#define deb(x) std::cout << #x << " = " << x << std::endl;

using namespace TNL;
using namespace TNL::Containers;

//-----------------------------------------------------------

__device__ void writeNewTask(int begin, int end, int depth
                             ArrayView<TASK, Devices::Cuda> newTasks, int *newTasksCnt,
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
                                      const Function &Cmp, int elemPerBlock, int depth,
                                      ArrayView<TASK, Devices::Cuda> tasks,
                                      ArrayView<int, Devices::Cuda> taskMapping, int *tasksAmount,
                                      ArrayView<TASK, Devices::Cuda> newTasks, int *newTasksCnt,
                                      ArrayView<TASK, Devices::Cuda> secondPhaseTasks, int *secondPhaseTasksCnt)
{
    static __shared__ int pivot;
    TASK &myTask = tasks[taskMapping[blockIdx.x]];

    if (threadIdx.x == 0)
        pivot = depth % 2 == 0 ? arr[myTask.partitionEnd - 1] : aux[myTask.partitionEnd - 1];
    __syncthreads();

    bool isLast;

    if (depth % 2 == 0)
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
                    depth,
                  newTasks, newTasksCnt,
                  secondPhaseTasks, secondPhaseTasksCnt);
}

//-----------------------------------------------------------

template <typename Function>
__global__ void cudaQuickSort2ndPhase(ArrayView<int, Devices::Cuda> arr, ArrayView<int, Devices::Cuda> aux,
                                      const Function &Cmp,
                                      ArrayView<TASK, Devices::Cuda> secondPhaseTasks)
{
    TASK & myTask = secondPhaseTasks[blockIdx.x];
    auto arrView = arr.getView(myTask.partitionBegin, myTask.partitionEnd);
    auto auxView = aux.getView(myTask.partitionBegin, myTask.partitionEnd);

    singleBlockQuickSort(arrView, auxView, Cmp, myTask.depth);
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
    int tasksAmount;                                                         //counter for Host == cuda_newTasksAmount
    int totalTask;                                                           // cuda_newTasksAmount + cuda_2ndPhaseTasksAmount

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
        cuda_tasks.setElement(0, TASK(0, arr.getSize()));
        totalTask = tasksAmount = 1;
        cuda_2ndPhaseTasksAmount = 0;
    }

    template <typename Function>
    void sort(const Function &Cmp)
    {
        while (tasksAmount > 0 && totalTask < maxTasks)
        {
            int elemPerBlock = getElemPerBlock();
            int blocksCnt = initTasks(elemPerBlock);

            if (iteration % 2 == 0)
            {
                cudaPartition<<<blocksCnt, threadsPerBlock>>>(
                    arr, Cmp,
                    aux.getView(),
                    cuda_blockToTaskMapping.getView(),
                    elemPerBlock,
                    cuda_tasks.getView(), cuda_newTasks.getView(),
                    cuda_newTasksAmount.getData(),
                    cuda_2ndPhaseTasks.getView(), cuda_2ndPhaseTasksAmount.getData());
            }
            else
            {
                cudaPartition<<<blocksCnt, threadsPerBlock>>>(
                    arr, Cmp,
                    aux.getView(),
                    cuda_blockToTaskMapping.getView(),
                    elemPerBlock,
                    cuda_newTasks.getView(), cuda_tasks.getView(), //swapped order to write back and forth without copying
                    cuda_newTasksAmount.getData(),
                    cuda_2ndPhaseTasks.getView(), cuda_2ndPhaseTasksAmount.getData());
            }

            tasksAmount = processNewTasks();

            iteration++;
        }

        _2ndPhase(Cmp);

        cudaDeviceSynchronize();
    }

    int getSetsNeeded() const
    {
        auto view = iteration % 2 == 0 ? cuda_tasks.getConstView() : cuda_newTasks.getConstView();
        auto fetch = [=] __cuda_callable__(int i) {
            auto &task = view[i];
            int size = task.arrEnd - task.arrBegin;
            return size / minElemPerBlock + (size % minElemPerBlock != 0);
        };
        auto reduction = [] __cuda_callable__(int a, int b) { return a + b; };
        return TNL::Algorithms::Reduction<TNL::Devices::Cuda>::reduce(0, tasksAmount, reduction, fetch, 0);
    }

    int getElemPerBlock() const
    {
        int setsNeeded = getSetsNeeded();

        if (setsNeeded <= maxBlocks)
            return minElemPerBlock;

        int setsPerBlock = setsNeeded / maxBlocks + 1; //+1 to spread out task of the last block
        return setsPerBlock * minElemPerBlock;
    }

    int initTasks(int elemPerBlock)
    {
        int threads = min(tasksAmount, threadsPerBlock);
        int blocks = tasksAmount / threads + (tasksAmount % threads != 0);
        cuda_blockToTaskMapping_Cnt = 0;

        if (iteration % 2 == 0)
        {
            cudaInitTask<<<blocks, threads>>>(
                cuda_tasks.getView(), tasksAmount, elemPerBlock,
                cuda_blockToTaskMapping_Cnt.getData(),
                cuda_blockToTaskMapping.getView());
        }
        else
        {
            cudaInitTask<<<blocks, threads>>>(
                cuda_newTasks.getView(), tasksAmount, elemPerBlock,
                cuda_blockToTaskMapping_Cnt.getData(),
                cuda_blockToTaskMapping.getView());
        }

        cuda_newTasksAmount.setElement(0, 0);
        return cuda_blockToTaskMapping_Cnt.getElement(0);
    }

    int processNewTasks()
    {
        TNL::Algorithms::MultiDeviceMemoryOperations<TNL::Devices::Cuda, TNL::Devices::Cuda>::
            copy(arr.getData(), aux.getData(), aux.getSize());

        tasksAmount = cuda_newTasksAmount.getElement(0);
        totalTask = tasksAmount + cuda_2ndPhaseTasksAmount.getElement(0);
        return tasksAmount;
    }

    template <typename Function>
    void _2ndPhase(const Function &Cmp)
    {
        if (totalTask == 0)
            return;

        TNL::Algorithms::MultiDeviceMemoryOperations<TNL::Devices::Cuda, TNL::Devices::Cuda>::
            copy(cuda_2ndPhaseTasks.getData() + (totalTask - tasksAmount),
                 (iteration % 2 ? cuda_newTasks.getData() : cuda_tasks.getData()),
                 tasksAmount);

        int blocks = totalTask;

        int stackSize = 128, stackMem = stackSize * sizeof(int);
        int bitonicMem = threadsPerBlock * 2 * sizeof(int);
        int auxMem = stackMem + bitonicMem;
        cudaQuickSort<<<blocks, threadsPerBlock, auxMem>>>(arr, Cmp, aux.getView(), stackSize, cuda_2ndPhaseTasks.getView());
    }
};

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
