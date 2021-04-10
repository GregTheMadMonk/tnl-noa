#pragma once

#include <TNL/Containers/Array.h>
#include "../util/reduction.cuh"
#include "task.h"
#include "cudaPartition.cuh"
#include "quicksort_1Block.cuh"

using namespace TNL;
using namespace TNL::Containers;

//-----------------------------------------------------------

__device__ void writeNewTask(int begin, int end, int depth, int maxElemFor2ndPhase,
                             ArrayView<TASK, Devices::Cuda> newTasks, int *newTasksCnt,
                             ArrayView<TASK, Devices::Cuda> secondPhaseTasks, int *secondPhaseTasksCnt);

//-----------------------------------------------------------

__global__ void cudaCalcBlocksNeeded(ArrayView<TASK, Devices::Cuda> cuda_tasks, int elemPerBlock,
                                     ArrayView<int, Devices::Cuda> blocksNeeded)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cuda_tasks.getSize())
        return;

    TASK &task = cuda_tasks[i];
    int size = task.partitionEnd - task.partitionBegin;
    blocksNeeded[i] = size / elemPerBlock + (size % elemPerBlock != 0);
}

//-----------------------------------------------------------

template <typename Value, typename Function>
__global__ void cudaInitTask(ArrayView<TASK, Devices::Cuda> cuda_tasks,
                             ArrayView<int, Devices::Cuda> cuda_blockToTaskMapping,
                             ArrayView<int, Devices::Cuda> cuda_reductionTaskInitMem,
                             ArrayView<Value, Devices::Cuda> src, const Function &Cmp)
{
    if (blockIdx.x >= cuda_tasks.getSize())
        return;

    int start = blockIdx.x == 0 ? 0 : cuda_reductionTaskInitMem[blockIdx.x - 1];
    int end = cuda_reductionTaskInitMem[blockIdx.x];
    for (int i = start + threadIdx.x; i < end; i += blockDim.x)
        cuda_blockToTaskMapping[i] = blockIdx.x;

    if (threadIdx.x == 0)
    {
        TASK &task = cuda_tasks[blockIdx.x];
        int pivotIdx = task.partitionBegin + pickPivotIdx(src.getView(task.partitionBegin, task.partitionEnd), Cmp);
        task.initTask(start, end - start, pivotIdx);
    }
}

//----------------------------------------------------

template <typename Value, typename Function, bool useShared>
__global__ void cudaQuickSort1stPhase(ArrayView<Value, Devices::Cuda> arr, ArrayView<Value, Devices::Cuda> aux,
                                      const Function &Cmp, int elemPerBlock,
                                      ArrayView<TASK, Devices::Cuda> tasks,
                                      ArrayView<int, Devices::Cuda> taskMapping)
{
    extern __shared__ int externMem[];
    Value *piv = (Value *)externMem;
    Value *sharedMem = piv + 1;

    TASK &myTask = tasks[taskMapping[blockIdx.x]];
    auto &src = (myTask.depth & 1) == 0 ? arr : aux;
    auto &dst = (myTask.depth & 1) == 0 ? aux : arr;

    if (threadIdx.x == 0)
        *piv = src[myTask.pivotIdx];
    __syncthreads();
    Value &pivot = *piv;

    cudaPartition<Value, Function, useShared>(
        src.getView(myTask.partitionBegin, myTask.partitionEnd),
        dst.getView(myTask.partitionBegin, myTask.partitionEnd),
        Cmp, sharedMem, pivot,
        elemPerBlock, myTask);
}

//----------------------------------------------------

template <typename Value>
__global__ void cudaWritePivot(ArrayView<Value, Devices::Cuda> arr, ArrayView<Value, Devices::Cuda> aux, int maxElemFor2ndPhase,
                               ArrayView<TASK, Devices::Cuda> tasks, ArrayView<TASK, Devices::Cuda> newTasks, int *newTasksCnt,
                               ArrayView<TASK, Devices::Cuda> secondPhaseTasks, int *secondPhaseTasksCnt)
{
    extern __shared__ int externMem[];
    Value *piv = (Value *)externMem;

    TASK &myTask = tasks[blockIdx.x];

    if (threadIdx.x == 0)
        *piv = (myTask.depth & 1) == 0 ? arr[myTask.pivotIdx] : aux[myTask.pivotIdx];
    __syncthreads();
    Value &pivot = *piv;

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

    if (leftEnd - leftBegin > 0)
    {
        writeNewTask(leftBegin, leftEnd, myTask.depth,
                     maxElemFor2ndPhase,
                     newTasks, newTasksCnt,
                     secondPhaseTasks, secondPhaseTasksCnt);
    }

    if (rightEnd - rightBegin > 0)
    {
        writeNewTask(rightBegin, rightEnd,
                     myTask.depth, maxElemFor2ndPhase,
                     newTasks, newTasksCnt,
                     secondPhaseTasks, secondPhaseTasksCnt);
    }
}

//-----------------------------------------------------------

__device__ void writeNewTask(int begin, int end, int depth, int maxElemFor2ndPhase,
                             ArrayView<TASK, Devices::Cuda> newTasks, int *newTasksCnt,
                             ArrayView<TASK, Devices::Cuda> secondPhaseTasks, int *secondPhaseTasksCnt)
{
    int size = end - begin;
    if (size < 0)
    {
        printf("negative size, something went really wrong\n");
        return;
    }

    if (size == 0)
        return;

    if (size <= maxElemFor2ndPhase)
    {
        int idx = atomicAdd(secondPhaseTasksCnt, 1);
        if (idx < secondPhaseTasks.getSize())
            secondPhaseTasks[idx] = TASK(begin, end, depth + 1);
        else
        {
            //printf("ran out of memory, trying backup\n");
            int idx = atomicAdd(newTasksCnt, 1);
            if (idx < newTasks.getSize())
                newTasks[idx] = TASK(begin, end, depth + 1);
            else
                printf("ran out of memory for second phase task, there isnt even space in newTask list\nPart of array may stay unsorted!!!\n");
        }
    }
    else
    {
        int idx = atomicAdd(newTasksCnt, 1);
        if (idx < newTasks.getSize())
            newTasks[idx] = TASK(begin, end, depth + 1);
        else
        {
            //printf("ran out of memory, trying backup\n");
            int idx = atomicAdd(secondPhaseTasksCnt, 1);
            if (idx < secondPhaseTasks.getSize())
                secondPhaseTasks[idx] = TASK(begin, end, depth + 1);
            else
                printf("ran out of memory for newtask, there isnt even space in second phase task list\nPart of array may stay unsorted!!!\n");
        }
    }
}

//-----------------------------------------------------------

template <typename Value, typename Function, int stackSize>
__global__ void cudaQuickSort2ndPhase(ArrayView<Value, Devices::Cuda> arr, ArrayView<Value, Devices::Cuda> aux,
                                      const Function &Cmp,
                                      ArrayView<TASK, Devices::Cuda> secondPhaseTasks,
                                      int elemInShared, int maxBitonicSize)
{
    extern __shared__ int externMem[];
    Value *sharedMem = (Value *)externMem;

    TASK &myTask = secondPhaseTasks[blockIdx.x];
    if (myTask.partitionEnd - myTask.partitionBegin <= 0)
    {
        //printf("empty task???\n");
        return;
    }

    auto arrView = arr.getView(myTask.partitionBegin, myTask.partitionEnd);
    auto auxView = aux.getView(myTask.partitionBegin, myTask.partitionEnd);

    if (elemInShared == 0)
    {
        singleBlockQuickSort<Value, Function, stackSize, false>
            (arrView, auxView, Cmp, myTask.depth, sharedMem, 0, maxBitonicSize);
    }
    else
    {
        singleBlockQuickSort<Value, Function, stackSize, true>
            (arrView, auxView, Cmp, myTask.depth, sharedMem, elemInShared, maxBitonicSize);
    }
}

template <typename Value, typename Function, int stackSize>
__global__ void cudaQuickSort2ndPhase(ArrayView<Value, Devices::Cuda> arr, ArrayView<Value, Devices::Cuda> aux,
                                      const Function &Cmp,
                                      ArrayView<TASK, Devices::Cuda> secondPhaseTasks1,
                                      ArrayView<TASK, Devices::Cuda> secondPhaseTasks2,
                                      int elemInShared, int maxBitonicSize)
{
    extern __shared__ int externMem[];
    Value *sharedMem = (Value *)externMem;

    TASK myTask;
    if (blockIdx.x < secondPhaseTasks1.getSize())
        myTask = secondPhaseTasks1[blockIdx.x];
    else
        myTask = secondPhaseTasks2[blockIdx.x - secondPhaseTasks1.getSize()];

    if (myTask.partitionEnd - myTask.partitionBegin <= 0)
    {
        //printf("empty task???\n");
        return;
    }

    auto arrView = arr.getView(myTask.partitionBegin, myTask.partitionEnd);
    auto auxView = aux.getView(myTask.partitionBegin, myTask.partitionEnd);

    if (elemInShared <= 0)
    {
        singleBlockQuickSort<Value, Function, stackSize, false>
            (arrView, auxView, Cmp, myTask.depth, sharedMem, 0, maxBitonicSize);
    }
    else
    {
        singleBlockQuickSort<Value, Function, stackSize, true>
            (arrView, auxView, Cmp, myTask.depth, sharedMem, elemInShared, maxBitonicSize);
    }
}

//-----------------------------------------------------------