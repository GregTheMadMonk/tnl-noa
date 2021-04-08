#pragma once

#include <TNL/Containers/Array.h>
#include "task.h"
#include "quicksort_kernel.cuh"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include <iostream>
#define deb(x) std::cout << #x << " = " << x << std::endl;

using namespace TNL;
using namespace TNL::Containers;

template <typename Value>
class QUICKSORT
{
    int maxBlocks, threadsPerBlock, desiredElemPerBlock, maxSharable; //kernel config

    //--------------------------------------

    Array<Value, Devices::Cuda> auxMem;
    ArrayView<Value, Devices::Cuda> arr, aux;

    //--------------------------------------

    const int maxBitonicSize = threadsPerBlock * 2;
    const int desired_2ndPhasElemPerBlock = maxBitonicSize;
    const int g_maxTasks = 1 << 14;
    int maxTasks;

    //--------------------------------------

    //cuda side task initialization and storing
    Array<TASK, Devices::Cuda> cuda_tasks, cuda_newTasks, cuda_2ndPhaseTasks; //1 set of 2 rotating tasks and 2nd phase
    Array<int, Devices::Cuda> cuda_newTasksAmount, cuda_2ndPhaseTasksAmount;  //is in reality 1 integer each

    Array<int, Devices::Cuda> cuda_blockToTaskMapping;
    Array<int, Devices::Cuda> cuda_reductionTaskInitMem;

    //--------------------------------------

    int host_1stPhaseTasksAmount = 0, host_2ndPhaseTasksAmount = 0;
    int iteration = 0;

    //--------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------
public:
    QUICKSORT(ArrayView<Value, Devices::Cuda> arr, int gridDim, int blockDim, int desiredElemPerBlock, int maxSharable)
        : maxBlocks(gridDim), threadsPerBlock(blockDim),
          desiredElemPerBlock(desiredElemPerBlock), maxSharable(maxSharable),

          arr(arr.getView()), auxMem(arr.getSize()), aux(auxMem.getView()),

          maxTasks(min(arr.getSize(), g_maxTasks)),

          cuda_tasks(maxTasks), cuda_newTasks(maxTasks), cuda_2ndPhaseTasks(maxTasks),
          cuda_newTasksAmount(1), cuda_2ndPhaseTasksAmount(1),

          cuda_blockToTaskMapping(maxBlocks),
          cuda_reductionTaskInitMem(maxTasks)
    {
        if (arr.getSize() > desired_2ndPhasElemPerBlock)
        {
            cuda_tasks.setElement(0, TASK(0, arr.getSize(), 0));
            host_1stPhaseTasksAmount = 1;
        }
        else
        {
            cuda_2ndPhaseTasks.setElement(0, TASK(0, arr.getSize(), 0));
            host_2ndPhaseTasksAmount = 1;
        }

        cuda_2ndPhaseTasksAmount = 0;
        TNL_CHECK_CUDA_DEVICE;
    }
    //--------------------------------------------------------------------------------------

    template <typename Function>
    void sort(const Function &Cmp);

    //--------------------------------------------------------------------------------------

    /**
     * returns how many blocks are needed to start sort phase 1 if @param elemPerBlock were to be used
     * */
    int getSetsNeeded(int elemPerBlock) const;

    /**
     * returns the optimal amount of elements per thread needed for phase 
     * */
    int getElemPerBlock() const;

    /**
     * returns the amount of blocks needed to start phase 1 while also initializing all tasks
     * */
    template <typename Function>
    int initTasks(int elemPerBlock, const Function &Cmp);

    /**
     * does the 1st phase of quicksort until out of task memory or each task is small enough
     * for correctness, secondphase method needs to be called to sort each subsequences
     * */
    template <typename Function>
    void firstPhase(const Function &Cmp);

    /**
     * update necessary variables after 1 phase1 sort
     * */
    void processNewTasks();

    /**
     * sorts all leftover tasks
     * */
    template <typename Function>
    void secondPhase(const Function &Cmp);
};

//---------------------------------------------------------------------------------------------
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

        /**
         * initializes tasks so that each block knows which task to work on and which part of array to split
         * also sets pivot needed for partitioning, this is why Cmp is needed
         * */
        int blocksCnt = initTasks(elemPerBlock, Cmp);
        TNL_CHECK_CUDA_DEVICE;

        //not enough or too many blocks needed, switch to 2nd phase
        if (blocksCnt <= 1 || blocksCnt > cuda_blockToTaskMapping.getSize())
            break;

        //-----------------------------------------------
        //do the partitioning

        auto &task = iteration % 2 == 0 ? cuda_tasks : cuda_newTasks;
        int externMemByteSize = elemPerBlock * sizeof(Value) + sizeof(Value); //elems + 1 for pivot

        /**
         * check if partition procedure can use shared memory for coalesced write after reordering
         * 
         * move elements smaller than pivot to the left and bigger to the right
         * note: pivot isnt inserted in the middle yet
         * */
        if (externMemByteSize <= maxSharable)
        {
            cudaQuickSort1stPhase<Value, Function, true>
                <<<blocksCnt, threadsPerBlock, externMemByteSize>>>(
                    arr, aux, Cmp, elemPerBlock,
                    task, cuda_blockToTaskMapping);
        }
        else
        {
            cudaQuickSort1stPhase<Value, Function, false>
                <<<blocksCnt, threadsPerBlock, sizeof(Value)>>>(
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
            <<<host_1stPhaseTasksAmount, threadsPerBlock, sizeof(Value)>>>(
                arr, aux, desired_2ndPhasElemPerBlock,
                task, newTask, cuda_newTasksAmount.getData(),
                cuda_2ndPhaseTasks, cuda_2ndPhaseTasksAmount.getData());
        TNL_CHECK_CUDA_DEVICE;

        //----------------------------------------

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

    int elemInShared = desiredElemPerBlock;
    int externSharedByteSize = elemInShared * sizeof(Value) + sizeof(Value); //reserve space for storing elements + 1 pivot
    if (externSharedByteSize > maxSharable)
    {
        externSharedByteSize = sizeof(Value);
        elemInShared = 0;
    }

    if (host_1stPhaseTasksAmount > 0 && host_2ndPhaseTasksAmount > 0)
    {
        auto tasks = leftoverTasks.getView(0, host_1stPhaseTasksAmount);
        auto tasks2 = cuda_2ndPhaseTasks.getView(0, host_2ndPhaseTasksAmount);

        cudaQuickSort2ndPhase<Value, Function, stackSize>
            <<<total2ndPhase, threadsPerBlock, externSharedByteSize>>>(arr, aux, Cmp, tasks, tasks2, elemInShared);
    }
    else if (host_1stPhaseTasksAmount > 0)
    {
        auto tasks = leftoverTasks.getView(0, host_1stPhaseTasksAmount);
        cudaQuickSort2ndPhase<Value, Function, stackSize>
            <<<total2ndPhase, threadsPerBlock, externSharedByteSize>>>(arr, aux, Cmp, tasks, elemInShared);
    }
    else
    {
        auto tasks2 = cuda_2ndPhaseTasks.getView(0, host_2ndPhaseTasksAmount);

        cudaQuickSort2ndPhase<Value, Function, stackSize>
            <<<total2ndPhase, threadsPerBlock, externSharedByteSize>>>(arr, aux, Cmp, tasks2, elemInShared);
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

    auto &src = iteration % 2 == 0 ? arr : aux;
    auto &tasks = iteration % 2 == 0 ? cuda_tasks : cuda_newTasks;

    //--------------------------------------------------------
    int blocks = host_1stPhaseTasksAmount / threadsPerBlock + (host_1stPhaseTasksAmount % threadsPerBlock != 0);

    cudaCalcBlocksNeeded<<<blocks, threadsPerBlock>>>(tasks.getView(0, host_1stPhaseTasksAmount), elemPerBlock,
                                                      cuda_reductionTaskInitMem.getView(0, host_1stPhaseTasksAmount));
    //cuda_reductionTaskInitMem[i] == how many blocks task i needs

    thrust::inclusive_scan(thrust::device,
                           cuda_reductionTaskInitMem.getData(),
                           cuda_reductionTaskInitMem.getData() + host_1stPhaseTasksAmount,
                           cuda_reductionTaskInitMem.getData());
    //cuda_reductionTaskInitMem[i] == how many blocks task [0..i] need

    int blocksNeeded = cuda_reductionTaskInitMem.getElement(host_1stPhaseTasksAmount - 1);

    //need too many blocks, give back control
    if (blocksNeeded > cuda_blockToTaskMapping.getSize())
        return blocksNeeded;

    //--------------------------------------------------------

    cudaInitTask<<<host_1stPhaseTasksAmount, threadsPerBlock>>>(
        tasks.getView(0, host_1stPhaseTasksAmount),                     //task to read from
        cuda_blockToTaskMapping.getView(0, blocksNeeded),               //maps block to a certain task
        cuda_reductionTaskInitMem.getView(0, host_1stPhaseTasksAmount), //has how many each task need blocks precalculated
        src, Cmp);                                                      //used to pick pivot

    cuda_newTasksAmount.setElement(0, 0); //resets new element counter
    return blocksNeeded;
}

template <typename Value>
void QUICKSORT<Value>::processNewTasks()
{
    host_1stPhaseTasksAmount = min(cuda_newTasksAmount.getElement(0), maxTasks);
    host_2ndPhaseTasksAmount = min(cuda_2ndPhaseTasksAmount.getElement(0), maxTasks);
}

//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------

template <typename Value, typename Function>
void quicksort(ArrayView<Value, Devices::Cuda> arr, const Function &Cmp)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    /**
     * for every block there is a bit of shared memory reserved, the actual value can slightly differ
     * */
    int sharedReserve = sizeof(int) * (16 + 3 * 32);
    int maxSharable = deviceProp.sharedMemPerBlock - sharedReserve;

    int blockDim = 512; //best case

    /**
     * the goal is to use shared memory as often as possible
     * each thread in a block will process n elements, n==multiplier
     * + 1 reserved for pivot (statically allocating Value type throws weird error, hence it needs to be dynamic)
     * 
     * blockDim*multiplier*sizeof(Value) + 1*sizeof(Value) <= maxSharable
     * */
    int elemPerBlock = (maxSharable - sizeof(Value)) / sizeof(Value); //try to use up all of shared memory to store elements
    const int maxBlocks = (1 << 20);
    const int maxMultiplier = 8;
    int multiplier = min(elemPerBlock / blockDim, maxMultiplier);

    if (multiplier <= 0) //a block cant store 512 elements, sorting some really big data
    {
        blockDim = 256; //try to fit 256 elements
        multiplier = min(elemPerBlock / blockDim, maxMultiplier);

        if (multiplier <= 0)
        {
            //worst case scenario, shared memory cant be utilized at all because of the sheer size of Value
            //sort has to be done with the use of global memory alone

            QUICKSORT<Value> sorter(arr, maxBlocks, 512, 0, 0);
            sorter.sort(Cmp);
            return;
        }
    }

    assert(blockDim * multiplier * sizeof(Value) <= maxSharable);

    QUICKSORT<Value> sorter(arr, maxBlocks, blockDim, multiplier * blockDim, maxSharable);
    sorter.sort(Cmp);
}

template <typename Value>
void quicksort(ArrayView<Value, Devices::Cuda> arr)
{
    quicksort(arr, [] __cuda_callable__(const Value &a, const Value &b) { return a < b; });
}
