#pragma once

#include <TNL/Containers/Array.h>
#include "../util/reduction.cuh"
#include "task.h"
#include "cudaPartition.cuh"
#include "quicksort_1Block.cuh"
#include "../bitonicSort/bitonicSort.h"
#include <iostream>

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#define deb(x) std::cout << #x << " = " << x << std::endl;

using namespace TNL;
using namespace TNL::Containers;

//-----------------------------------------------------------

__device__ void writeNewTask(int begin, int end, int depth, int maxElemFor2ndPhase,
                            ArrayView<TASK, Devices::Cuda> newTasks, int *newTasksCnt,
                             ArrayView<TASK, Devices::Cuda> secondPhaseTasks, int *secondPhaseTasksCnt)
{
    int size = end - begin;
    if(size < 0)
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

//----------------------------------------------------

template <typename Function>
__global__ void cudaQuickSort1stPhase_1(ArrayView<int, Devices::Cuda> arr, ArrayView<int, Devices::Cuda> aux,
                                      const Function &Cmp, int elemPerBlock,
                                      ArrayView<TASK, Devices::Cuda> tasks,
                                      ArrayView<int, Devices::Cuda> taskMapping)
{
    extern __shared__ int externMem[];
    int *sharedMem = externMem;

    static __shared__ int pivot;

    TASK &myTask = tasks[taskMapping[blockIdx.x]];
    auto & src = (myTask.depth & 1) == 0? arr : aux;
    auto & dst = (myTask.depth & 1) == 0? aux : arr;

    if (threadIdx.x == 0)
        pivot = src[myTask.pivotIdx];
    __syncthreads();

    cudaPartition_1(
        src.getView(myTask.partitionBegin, myTask.partitionEnd),
        dst.getView(myTask.partitionBegin, myTask.partitionEnd),
        sharedMem,
        Cmp, pivot, elemPerBlock, myTask);
}

template <typename Function>
__global__ void cudaQuickSort1stPhase_2(ArrayView<int, Devices::Cuda> arr, ArrayView<int, Devices::Cuda> aux,
                                      const Function &Cmp, int elemPerBlock,
                                      ArrayView<TASK, Devices::Cuda> tasks,
                                      ArrayView<int, Devices::Cuda> taskMapping)
{
    static __shared__ int pivot;

    TASK &myTask = tasks[taskMapping[blockIdx.x]];
    auto & src = (myTask.depth & 1) == 0? arr : aux;
    auto & dst = (myTask.depth & 1) == 0? aux : arr;

    if (threadIdx.x == 0)
        pivot = src[myTask.pivotIdx];
    __syncthreads();

    cudaPartition_2(
        src.getView(myTask.partitionBegin, myTask.partitionEnd),
        dst.getView(myTask.partitionBegin, myTask.partitionEnd),
        Cmp, pivot, elemPerBlock, myTask);
}

//----------------------------------------------------


__global__ void cudaWritePivot(ArrayView<int, Devices::Cuda> arr, ArrayView<int, Devices::Cuda> aux, int maxElemFor2ndPhase,
                               ArrayView<TASK, Devices::Cuda> tasks, ArrayView<TASK, Devices::Cuda> newTasks, int *newTasksCnt,
                               ArrayView<TASK, Devices::Cuda> secondPhaseTasks, int *secondPhaseTasksCnt)
{
    static __shared__ int pivot;
    TASK &myTask = tasks[blockIdx.x];

    if (threadIdx.x == 0)
    {
        if ((myTask.depth & 1) == 0)
            pivot = arr[myTask.pivotIdx];
        else
            pivot = aux[myTask.pivotIdx];
    }
    __syncthreads();

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

    if(leftEnd - leftBegin > 0)
    {
        writeNewTask(leftBegin, leftEnd, myTask.depth,
                    maxElemFor2ndPhase,
                    newTasks, newTasksCnt,
                    secondPhaseTasks, secondPhaseTasksCnt);
    }

    if(rightEnd - rightBegin > 0)
    {
        writeNewTask(rightBegin, rightEnd,
                    myTask.depth, maxElemFor2ndPhase,
                    newTasks, newTasksCnt,
                    secondPhaseTasks, secondPhaseTasksCnt);
    }
}

//-----------------------------------------------------------

template <typename Function, int stackSize>
__global__ void cudaQuickSort2ndPhase(ArrayView<int, Devices::Cuda> arr, ArrayView<int, Devices::Cuda> aux,
                                      const Function &Cmp,
                                      ArrayView<TASK, Devices::Cuda> secondPhaseTasks)
{
    TASK &myTask = secondPhaseTasks[blockIdx.x];
    if(myTask.partitionEnd - myTask.partitionBegin <= 0 )
        return;

    auto arrView = arr.getView(myTask.partitionBegin, myTask.partitionEnd);
    auto auxView = aux.getView(myTask.partitionBegin, myTask.partitionEnd);

    singleBlockQuickSort<Function, stackSize>(arrView, auxView, Cmp, myTask.depth);
}

//-----------------------------------------------------------

__global__ void cudaCalcBlocksNeeded(ArrayView<TASK, Devices::Cuda> cuda_tasks, int elemPerBlock,
                                    ArrayView<int, Devices::Cuda> blocksNeeded)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= cuda_tasks.getSize())
        return;

    auto task = cuda_tasks[i];
    int size = task.partitionEnd - task.partitionBegin;
    blocksNeeded[i] = size / elemPerBlock + (size % elemPerBlock != 0);
}                                    

template <typename Function>
__global__ void cudaInitTask(ArrayView<TASK, Devices::Cuda> cuda_tasks,
                             ArrayView<int, Devices::Cuda> cuda_blockToTaskMapping,
                             ArrayView<int, Devices::Cuda> cuda_reductionTaskInitMem,
                             ArrayView<int, Devices::Cuda> src, const Function &Cmp)
{
    if(blockIdx.x >= cuda_tasks.getSize())
        return;

    int start = blockIdx.x == 0? 0 : cuda_reductionTaskInitMem[blockIdx.x -1];
    int end = cuda_reductionTaskInitMem[blockIdx.x];
    for(int i = start + threadIdx.x; i < end; i += blockDim.x)
        cuda_blockToTaskMapping[i] = blockIdx.x;

    if(threadIdx.x == 0)
    {
        TASK & task = cuda_tasks[blockIdx.x];
        int pivotIdx = task.partitionBegin + pickPivotIdx(src.getView(task.partitionBegin, task.partitionEnd), Cmp);
        task.initTask(start, end-start, pivotIdx);
    }
}
//-----------------------------------------------------------
//-----------------------------------------------------------
const int threadsPerBlock = 512, g_maxBlocks = 1 << 15; //32k
const int g_maxTasks = 1 << 14;
const int minElemPerBlock = threadsPerBlock*2;
const int maxBitonicSize = threadsPerBlock*2;
const int desired_2ndPhasElemPerBlock = maxBitonicSize*8;

class QUICKSORT
{
    ArrayView<int, Devices::Cuda> arr;
    Array<int, Devices::Cuda> aux;
    int maxTasks, maxBlocks;
    Array<TASK, Devices::Cuda> cuda_tasks, cuda_newTasks, cuda_2ndPhaseTasks;

    Array<int, Devices::Cuda> cuda_newTasksAmount, cuda_2ndPhaseTasksAmount; //is in reality 1 integer

    int tasksAmount;              //counter for Host == cuda_newTasksAmount
    int host_2ndPhaseTasksAmount; // cuda_2ndPhaseTasksAmount

    Array<int, Devices::Cuda> cuda_blockToTaskMapping;
    Array<int, Devices::Cuda> cuda_reductionTaskInitMem;

    int iteration = 0;
    //--------------------------------------------------------------------------------------
    cudaDeviceProp deviceProp;
    //--------------------------------------------------------------------------------------
public:
    QUICKSORT(ArrayView<int, Devices::Cuda> _arr)
        : arr(_arr), aux(arr.getSize()),
          maxTasks(min(arr.getSize(), g_maxTasks)),
          maxBlocks(g_maxBlocks),
          cuda_tasks(maxTasks), cuda_newTasks(maxTasks), cuda_2ndPhaseTasks(maxTasks),
          cuda_newTasksAmount(1),
          cuda_2ndPhaseTasksAmount(1),
          cuda_blockToTaskMapping(maxBlocks),
          cuda_reductionTaskInitMem(maxTasks)
    {
        cuda_tasks.setElement(0, TASK(0, arr.getSize(), 0));
        tasksAmount = 1;
        host_2ndPhaseTasksAmount = 0;
        cuda_2ndPhaseTasksAmount = 0;
        iteration = 0;

        cudaGetDeviceProperties(&deviceProp, 0); //change device
        TNL_CHECK_CUDA_DEVICE;
    }

    template <typename Function>
    void sort(const Function &Cmp);

    int getSetsNeeded(int elemPerBlock) const;
    int getElemPerBlock() const;

    /**
     * returns the amount of blocks needed
     * */
    template <typename Function>
    int initTasks(int elemPerBlock, const Function & Cmp);

    void processNewTasks();
};

template <typename Function>
void QUICKSORT::sort(const Function &Cmp)
{
    
    while (tasksAmount > 0)
    {
        //2ndphase task is now full or tasksAmount is full, as backup during writing, overflowing tasks were written into the other array
        if (tasksAmount >= maxTasks || host_2ndPhaseTasksAmount >= maxTasks)
        {
            break;
        }

        //just in case newly created tasks wouldnt fit
        if(tasksAmount*2 >= maxTasks + (maxTasks - host_2ndPhaseTasksAmount))
        {
            break;
        }

        int elemPerBlock = getElemPerBlock();
        int blocksCnt = initTasks(elemPerBlock, Cmp);
        if(blocksCnt >= cuda_blockToTaskMapping.getSize())
            break;

        TNL_CHECK_CUDA_DEVICE;

        int externMemByteSize = elemPerBlock * sizeof(int);
        auto & task = iteration % 2 == 0? cuda_tasks : cuda_newTasks;

        if(externMemByteSize <= deviceProp.sharedMemPerBlock)
        {
            cudaQuickSort1stPhase_1<Function>
                <<<blocksCnt, threadsPerBlock, externMemByteSize>>>(
                    arr, aux, Cmp, elemPerBlock,
                    task, cuda_blockToTaskMapping);
        }
        else
        {
            cudaQuickSort1stPhase_2<Function>
                <<<blocksCnt, threadsPerBlock>>>(
                    arr, aux, Cmp, elemPerBlock,
                task, cuda_blockToTaskMapping);
        }
                
        TNL_CHECK_CUDA_DEVICE;

        auto & newTask = iteration % 2 == 0? cuda_newTasks : cuda_tasks;
        cudaWritePivot<<<tasksAmount, 1024>>>(
            arr, aux, desired_2ndPhasElemPerBlock,
            task, newTask, cuda_newTasksAmount.getData(),
            cuda_2ndPhaseTasks, cuda_2ndPhaseTasksAmount.getData());

        TNL_CHECK_CUDA_DEVICE;

        processNewTasks();
        iteration++;
    }
    
    if (tasksAmount > 0)
    {
        auto & tasks = iteration % 2 == 0 ? cuda_tasks : cuda_newTasks;
        cudaQuickSort2ndPhase<Function, 128>
            <<<min(tasksAmount,tasks.getSize()) , threadsPerBlock>>>(arr, aux, Cmp, tasks);

        TNL_CHECK_CUDA_DEVICE;
        cudaDeviceSynchronize();
        TNL_CHECK_CUDA_DEVICE;
    }
    
    if (host_2ndPhaseTasksAmount > 0)
    {
        cudaQuickSort2ndPhase<Function, 128>
            <<<min(host_2ndPhaseTasksAmount,cuda_2ndPhaseTasks.getSize()) , threadsPerBlock>>>
            (arr, aux, Cmp, cuda_2ndPhaseTasks);

        TNL_CHECK_CUDA_DEVICE;
    }
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    return;
}

int QUICKSORT::getSetsNeeded(int elemPerBlock) const
{
    auto view = iteration % 2 == 0 ? cuda_tasks.getConstView() : cuda_newTasks.getConstView();
    auto fetch = [=] __cuda_callable__(int i) {
        auto &task = view[i];
        int size = task.partitionEnd - task.partitionBegin;
        return size / elemPerBlock + (size % elemPerBlock != 0);
    };
    auto reduction = [] __cuda_callable__(int a, int b) { return a + b; };
    return Algorithms::Reduction<Devices::Cuda>::reduce(0, tasksAmount, fetch, reduction, 0);
}

int QUICKSORT::getElemPerBlock() const
{
    int setsNeeded = getSetsNeeded(minElemPerBlock);

    if (setsNeeded <= maxBlocks)
        return minElemPerBlock;

    int setsPerBlock = ceil(1. * setsNeeded / maxBlocks);
    return setsPerBlock * minElemPerBlock;
}

template <typename Function>
int QUICKSORT::initTasks(int elemPerBlock, const Function & Cmp)
{
    int threads = min(tasksAmount, threadsPerBlock);
    int blocks = tasksAmount / threads + (tasksAmount % threads != 0);

    auto src = iteration % 2 == 0? arr : aux.getView();
    auto &tasks = iteration % 2 == 0? cuda_tasks : cuda_newTasks;

    //[i] == how many blocks task i needs
    cudaCalcBlocksNeeded<<<threads, blocks>>>(tasks.getView(0, tasksAmount),
        elemPerBlock, cuda_reductionTaskInitMem.getView(0, tasksAmount));

    thrust::inclusive_scan(thrust::device,
        cuda_reductionTaskInitMem.getData(),
        cuda_reductionTaskInitMem.getData() + tasksAmount,
        cuda_reductionTaskInitMem.getData());

    int blocksNeeded = cuda_reductionTaskInitMem.getElement(tasksAmount - 1);
    //need too many blocks, give back control
    if(blocksNeeded >= cuda_blockToTaskMapping.getSize())
        return blocksNeeded;

    cudaInitTask<<<tasksAmount, 512>>>(
        tasks.getView(0, tasksAmount),
        cuda_blockToTaskMapping.getView(0, blocksNeeded),
        cuda_reductionTaskInitMem.getView(0, tasksAmount),
        src, Cmp
    );

    cuda_newTasksAmount.setElement(0, 0);
    return blocksNeeded;
}

void QUICKSORT::processNewTasks()
{
    tasksAmount = cuda_newTasksAmount.getElement(0);
    host_2ndPhaseTasksAmount = cuda_2ndPhaseTasksAmount.getElement(0);
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
