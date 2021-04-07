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

//----------------------------------------------------

template <typename Value, typename Function>
__global__ void cudaQuickSort1stPhase_1(ArrayView<Value, Devices::Cuda> arr, ArrayView<Value, Devices::Cuda> aux,
                                        const Function &Cmp, int elemPerBlock,
                                        ArrayView<TASK, Devices::Cuda> tasks,
                                        ArrayView<int, Devices::Cuda> taskMapping)
{
    extern __shared__ int externMem[];
    Value *sharedMem = (Value *)externMem;

    static __shared__ Value pivot;

    TASK &myTask = tasks[taskMapping[blockIdx.x]];
    auto &src = (myTask.depth & 1) == 0 ? arr : aux;
    auto &dst = (myTask.depth & 1) == 0 ? aux : arr;

    if (threadIdx.x == 0)
        pivot = src[myTask.pivotIdx];
    __syncthreads();

    cudaPartition_1(
        src.getView(myTask.partitionBegin, myTask.partitionEnd),
        dst.getView(myTask.partitionBegin, myTask.partitionEnd),
        sharedMem,
        Cmp, pivot, elemPerBlock, myTask);
}

template <typename Value, typename Function>
__global__ void cudaQuickSort1stPhase_2(ArrayView<Value, Devices::Cuda> arr, ArrayView<Value, Devices::Cuda> aux,
                                        const Function &Cmp, int elemPerBlock,
                                        ArrayView<TASK, Devices::Cuda> tasks,
                                        ArrayView<int, Devices::Cuda> taskMapping)
{
    static __shared__ Value pivot;

    TASK &myTask = tasks[taskMapping[blockIdx.x]];
    auto &src = (myTask.depth & 1) == 0 ? arr : aux;
    auto &dst = (myTask.depth & 1) == 0 ? aux : arr;

    if (threadIdx.x == 0)
        pivot = src[myTask.pivotIdx];
    __syncthreads();

    cudaPartition_2(
        src.getView(myTask.partitionBegin, myTask.partitionEnd),
        dst.getView(myTask.partitionBegin, myTask.partitionEnd),
        Cmp, pivot, elemPerBlock, myTask);
}

//----------------------------------------------------

template <typename Value>
__global__ void cudaWritePivot(ArrayView<Value, Devices::Cuda> arr, ArrayView<Value, Devices::Cuda> aux, int maxElemFor2ndPhase,
                               ArrayView<TASK, Devices::Cuda> tasks, ArrayView<TASK, Devices::Cuda> newTasks, int *newTasksCnt,
                               ArrayView<TASK, Devices::Cuda> secondPhaseTasks, int *secondPhaseTasksCnt)
{
    static __shared__ Value pivot;
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

template <typename Value, typename Function, int stackSize>
__global__ void cudaQuickSort2ndPhase(ArrayView<Value, Devices::Cuda> arr, ArrayView<Value, Devices::Cuda> aux,
                                      const Function &Cmp,
                                      ArrayView<TASK, Devices::Cuda> secondPhaseTasks)
{
    TASK &myTask = secondPhaseTasks[blockIdx.x];
    if (myTask.partitionEnd - myTask.partitionBegin <= 0)
        return;

    auto arrView = arr.getView(myTask.partitionBegin, myTask.partitionEnd);
    auto auxView = aux.getView(myTask.partitionBegin, myTask.partitionEnd);

    singleBlockQuickSort<Value, Function, stackSize>(arrView, auxView, Cmp, myTask.depth);
}

template <typename Value, typename Function, int stackSize>
__global__ void cudaQuickSort2ndPhase(ArrayView<Value, Devices::Cuda> arr, ArrayView<Value, Devices::Cuda> aux,
                                      const Function &Cmp,
                                      ArrayView<TASK, Devices::Cuda> secondPhaseTasks1,
                                      ArrayView<TASK, Devices::Cuda> secondPhaseTasks2)
{
    TASK myTask;
    if (blockIdx.x < secondPhaseTasks1.getSize())
        myTask = secondPhaseTasks1[blockIdx.x];
    else
        myTask = secondPhaseTasks2[blockIdx.x - secondPhaseTasks1.getSize()];

    if (myTask.partitionEnd - myTask.partitionBegin <= 0)
        return;

    auto arrView = arr.getView(myTask.partitionBegin, myTask.partitionEnd);
    auto auxView = aux.getView(myTask.partitionBegin, myTask.partitionEnd);

    singleBlockQuickSort<Value, Function, stackSize>(arrView, auxView, Cmp, myTask.depth);
}

//-----------------------------------------------------------

__global__ void cudaCalcBlocksNeeded(ArrayView<TASK, Devices::Cuda> cuda_tasks, int elemPerBlock,
                                     ArrayView<int, Devices::Cuda> blocksNeeded)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cuda_tasks.getSize())
        return;

    auto task = cuda_tasks[i];
    int size = task.partitionEnd - task.partitionBegin;
    blocksNeeded[i] = size / elemPerBlock + (size % elemPerBlock != 0);
}

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
//-----------------------------------------------------------
//-----------------------------------------------------------

template <typename Value>
class QUICKSORT
{
    ArrayView<Value, Devices::Cuda> arr;
    Array<Value, Devices::Cuda> aux;

    int maxBlocks, threadsPerBlock, desiredElemPerBlock, maxSharable;

    const int maxBitonicSize = threadsPerBlock * 2;
    const int desired_2ndPhasElemPerBlock = maxBitonicSize;
    const int g_maxTasks = 1 << 14;

    int maxTasks;
    Array<TASK, Devices::Cuda> cuda_tasks, cuda_newTasks, cuda_2ndPhaseTasks;

    Array<int, Devices::Cuda> cuda_newTasksAmount, cuda_2ndPhaseTasksAmount; //is in reality 1 integer each

    int host_1stPhaseTasksAmount; //counter for Host == cuda_newTasksAmount
    int host_2ndPhaseTasksAmount; // cuda_2ndPhaseTasksAmount

    Array<int, Devices::Cuda> cuda_blockToTaskMapping;
    Array<int, Devices::Cuda> cuda_reductionTaskInitMem;

    int iteration = 0;
    //--------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------
public:
    QUICKSORT(ArrayView<Value, Devices::Cuda> arr, int gridDim, int blockDim, int desiredElemPerBlock, int maxSharable)
        : arr(arr.getView()), aux(arr.getSize()),
          maxBlocks(gridDim), threadsPerBlock(blockDim),
          desiredElemPerBlock(desiredElemPerBlock), maxSharable(maxSharable),

          maxTasks(min(arr.getSize(), g_maxTasks)),

          cuda_tasks(maxTasks), cuda_newTasks(maxTasks), cuda_2ndPhaseTasks(maxTasks),
          cuda_newTasksAmount(1), cuda_2ndPhaseTasksAmount(1),

          cuda_blockToTaskMapping(maxBlocks),
          cuda_reductionTaskInitMem(maxTasks)
    {
        cuda_tasks.setElement(0, TASK(0, arr.getSize(), 0));
        host_1stPhaseTasksAmount = 1;

        host_2ndPhaseTasksAmount = 0;
        cuda_2ndPhaseTasksAmount = 0;
        iteration = 0;

        TNL_CHECK_CUDA_DEVICE;
    }

    template <typename Function>
    void sort(const Function &Cmp);

    template <typename Function>
    void firstPhase(const Function &Cmp);

    template <typename Function>
    void secondPhase(const Function &Cmp);

    int getSetsNeeded(int elemPerBlock) const;
    int getElemPerBlock() const;

    /**
     * returns the amount of blocks needed
     * */
    template <typename Function>
    int initTasks(int elemPerBlock, const Function &Cmp);

    void processNewTasks();
};

//---------------------------------------------------------------------------------------------

template <typename Value>
template <typename Function>
void QUICKSORT<Value>::sort(const Function &Cmp)
{
    firstPhase(Cmp);

    int total2ndPhase = host_1stPhaseTasksAmount + host_2ndPhaseTasksAmount;
    if (total2ndPhase > 0)
        secondPhase(Cmp);

    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    return;
}

//---------------------------------------------------------------------------------------------

template <typename Value>
template <typename Function>
void QUICKSORT<Value>::firstPhase(const Function &Cmp)
{
    while (host_1stPhaseTasksAmount > 0)
    {
        //2ndphase task is now full or host_1stPhaseTasksAmount is full, as backup during writing, overflowing tasks were written into the other array
        if (host_1stPhaseTasksAmount >= maxTasks || host_2ndPhaseTasksAmount >= maxTasks)
            break;

        //just in case newly created tasks wouldnt fit
        if (host_1stPhaseTasksAmount * 2 >= maxTasks + (maxTasks - host_2ndPhaseTasksAmount))
            break;

        int elemPerBlock = getElemPerBlock();
        int blocksCnt = initTasks(elemPerBlock, Cmp);
        TNL_CHECK_CUDA_DEVICE;

        if (blocksCnt >= maxBlocks) //too many blocks needed, switch to 2nd phase
            break;

        //-----------------------------------------------
        //do the partitioning

        auto &task = iteration % 2 == 0 ? cuda_tasks : cuda_newTasks;
        int externMemByteSize = elemPerBlock * sizeof(Value);

        /**
         * check if can partition using shared memory for coalesced read and write
         * 1st phase of partitioning
         * sets of blocks work on a task
         * 
         * using the atomicAdd intristic, each block reserves a chunk of memory where to move elements
         * smaller and bigger than pivot move to
         * */
        if (externMemByteSize <= maxSharable)
        {
            cudaQuickSort1stPhase_1<Value, Function>
                <<<blocksCnt, threadsPerBlock, externMemByteSize>>>(
                    arr, aux, Cmp, elemPerBlock,
                    task, cuda_blockToTaskMapping);
        }
        else
        {
            cudaQuickSort1stPhase_2<Value, Function>
                <<<blocksCnt, threadsPerBlock>>>(
                    arr, aux, Cmp, elemPerBlock,
                    task, cuda_blockToTaskMapping);
        }
        TNL_CHECK_CUDA_DEVICE;

        /**
         * fill in the gap between smaller and bigger with elements == pivot
         * after writing also create new tasks, each task generates at max 2 tasks
         * 
         * tasks smaller than desired_2ndPhasElemPerBlock go into 2nd phase
         * bigger need more blocks to partition and are written into newTask
         * with iteration %2, rotate between the 2 tasks array to save from copying
         * */
        auto &newTask = iteration % 2 == 0 ? cuda_newTasks : cuda_tasks;
        cudaWritePivot<Value>
            <<<host_1stPhaseTasksAmount, 1024>>>(
                arr, aux, desired_2ndPhasElemPerBlock,
                task, newTask, cuda_newTasksAmount.getData(),
                cuda_2ndPhaseTasks, cuda_2ndPhaseTasksAmount.getData());
        TNL_CHECK_CUDA_DEVICE;

        processNewTasks();
        iteration++;
    }
}

//----------------------------------------------------------------------

template <typename Value>
template <typename Function>
void QUICKSORT<Value>::secondPhase(const Function &Cmp)
{
    int total2ndPhase = host_1stPhaseTasksAmount + host_2ndPhaseTasksAmount;
    const int stackSize = 32;
    auto &leftoverTasks = iteration % 2 == 0 ? cuda_tasks : cuda_newTasks;

    if (host_1stPhaseTasksAmount > 0 && host_2ndPhaseTasksAmount > 0)
    {
        auto tasks2 = cuda_2ndPhaseTasks.getView(0, host_2ndPhaseTasksAmount);

        cudaQuickSort2ndPhase<Value, Function, stackSize>
            <<<total2ndPhase, threadsPerBlock>>>(arr, aux, Cmp, leftoverTasks, tasks2);
    }
    else if (host_1stPhaseTasksAmount > 0)
    {
        auto tasks = leftoverTasks.getView(0, host_1stPhaseTasksAmount);
        cudaQuickSort2ndPhase<Value, Function, stackSize>
            <<<total2ndPhase, threadsPerBlock>>>(arr, aux, Cmp, tasks);
    }
    else
    {
        auto tasks2 = cuda_2ndPhaseTasks.getView(0, host_2ndPhaseTasksAmount);

        cudaQuickSort2ndPhase<Value, Function, stackSize>
            <<<total2ndPhase, threadsPerBlock>>>(arr, aux, Cmp, tasks2);
    }
}

//----------------------------------------------------------------------

template <typename Value>
int QUICKSORT<Value>::getSetsNeeded(int elemPerBlock) const
{
    auto view = iteration % 2 == 0 ? cuda_tasks.getConstView() : cuda_newTasks.getConstView();
    auto fetch = [=] __cuda_callable__(int i) {
        const auto &task = view[i];
        int size = task.partitionEnd - task.partitionBegin;
        return size / elemPerBlock + (size % elemPerBlock != 0);
    };
    auto reduction = [] __cuda_callable__(int a, int b) { return a + b; };
    return Algorithms::Reduction<Devices::Cuda>::reduce(0, host_1stPhaseTasksAmount, fetch, reduction, 0);
}

template <typename Value>
int QUICKSORT<Value>::getElemPerBlock() const
{
    int setsNeeded = getSetsNeeded(desiredElemPerBlock);

    if (setsNeeded <= maxBlocks)
        return desiredElemPerBlock;

    //want multiplier*minElemPerBLock <= x*threadPerBlock
    //find smallest x so that this inequality holds
    double multiplier = 1. * setsNeeded / maxBlocks;
    int elemPerBlock = multiplier * desiredElemPerBlock;
    setsNeeded = elemPerBlock / threadsPerBlock + (elemPerBlock % threadsPerBlock != 0);

    return setsNeeded * threadsPerBlock;
}

template <typename Value>
template <typename Function>
int QUICKSORT<Value>::initTasks(int elemPerBlock, const Function &Cmp)
{
    int threads = min(host_1stPhaseTasksAmount, threadsPerBlock);
    int blocks = host_1stPhaseTasksAmount / threads + (host_1stPhaseTasksAmount % threads != 0);

    auto src = iteration % 2 == 0 ? arr : aux.getView();
    auto &tasks = iteration % 2 == 0 ? cuda_tasks : cuda_newTasks;

    //[i] == how many blocks task i needs
    cudaCalcBlocksNeeded<<<threads, blocks>>>(tasks.getView(0, host_1stPhaseTasksAmount),
                                              elemPerBlock, cuda_reductionTaskInitMem.getView(0, host_1stPhaseTasksAmount));

    thrust::inclusive_scan(thrust::device,
                           cuda_reductionTaskInitMem.getData(),
                           cuda_reductionTaskInitMem.getData() + host_1stPhaseTasksAmount,
                           cuda_reductionTaskInitMem.getData());

    int blocksNeeded = cuda_reductionTaskInitMem.getElement(host_1stPhaseTasksAmount - 1);
    //need too many blocks, give back control
    if (blocksNeeded >= cuda_blockToTaskMapping.getSize())
        return blocksNeeded;

    cudaInitTask<<<host_1stPhaseTasksAmount, 512>>>(
        tasks.getView(0, host_1stPhaseTasksAmount),
        cuda_blockToTaskMapping.getView(0, blocksNeeded),
        cuda_reductionTaskInitMem.getView(0, host_1stPhaseTasksAmount),
        src, Cmp);

    cuda_newTasksAmount.setElement(0, 0);
    return blocksNeeded;
}

template <typename Value>
void QUICKSORT<Value>::processNewTasks()
{
    host_1stPhaseTasksAmount = cuda_newTasksAmount.getElement(0);
    host_2ndPhaseTasksAmount = cuda_2ndPhaseTasksAmount.getElement(0);
}

//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------

template <typename Value, typename Function>
void quicksort(ArrayView<Value, Devices::Cuda> arr, const Function &Cmp)
{
    const int maxBlocks = (1 << 20);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int sharedReserve = sizeof(Value) + sizeof(int) * 16; //1pivot + 16 other shared vars reserved
    int maxSharable = deviceProp.sharedMemPerBlock - sharedReserve;

    //blockDim*multiplier*sizeof(Value) <= maxSharable

    int blockDim = 512; //best case
    int elemPerBlock = maxSharable / sizeof(Value);
    const int maxMultiplier = 8;
    int multiplier = min(elemPerBlock / blockDim, maxMultiplier);
    if (multiplier <= 0)
    {
        blockDim = 256;
        multiplier = min(elemPerBlock / blockDim, maxMultiplier);
        if (multiplier <= 0)
        {
            //worst case scenario, shared memory cant be utilized at all because of the sheer size of Value
            //sort has to be done with the use of global memory alone

            QUICKSORT<Value> sorter(arr, maxBlocks, 512, 0, maxSharable);
            sorter.sort(Cmp);
            return;
        }
    }

    assert(blockDim * multiplier * sizeof(Value) <= maxSharable);

    QUICKSORT<Value> sorter(arr, maxBlocks, blockDim, multiplier*blockDim, maxSharable);
    sorter.sort(Cmp);
}

template <typename Value>
void quicksort(ArrayView<Value, Devices::Cuda> arr)
{
    quicksort(arr, [] __cuda_callable__(const Value &a, const Value &b) { return a < b; });
}
