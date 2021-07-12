#pragma once

#include <TNL/Containers/Array.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Scan.h>
#include <TNL/Algorithms/detail/Sorting/task.h>
#include <TNL/Algorithms/detail/Sorting/quicksort_kernel.cuh>

#include <iostream>
#define deb(x) std::cout << #x << " = " << x << std::endl;

#ifdef CHECK_RESULT_SORT
#include "../util/algorithm.h"
#include <fstream>
#endif

using namespace TNL;
using namespace TNL::Containers;

namespace TNL {
    namespace Algorithms {
        namespace detail {

template <typename Value>
class QUICKSORT
{
    int maxBlocks, threadsPerBlock, desiredElemPerBlock, maxSharable; //kernel config

    //--------------------------------------

    Array<Value, Devices::Cuda> auxMem;
    ArrayView<Value, Devices::Cuda> arr, aux;

    //--------------------------------------

    int desired_2ndPhasElemPerBlock;
    const int g_maxTasks = 1 << 14;
    int maxTasks;

    //--------------------------------------

    //cuda side task initialization and storing
    Array<TASK, Devices::Cuda> cuda_tasks, cuda_newTasks, cuda_2ndPhaseTasks; //1 set of 2 rotating tasks and 2nd phase
    Array<int, Devices::Cuda> cuda_newTasksAmount, cuda_2ndPhaseTasksAmount;  //is in reality 1 integer each

    Array<int, Devices::Cuda> cuda_blockToTaskMapping;
    Vector<int, Devices::Cuda> cuda_reductionTaskInitMem;

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

          desired_2ndPhasElemPerBlock(desiredElemPerBlock),
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

    template <typename CMP>
    void sort(const CMP &Cmp);

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
    template <typename CMP>
    int initTasks(int elemPerBlock, const CMP &Cmp);

    /**
     * does the 1st phase of quicksort until out of task memory or each task is small enough
     * for correctness, secondphase method needs to be called to sort each subsequences
     * */
    template <typename CMP>
    void firstPhase(const CMP &Cmp);

    /**
     * update necessary variables after 1 phase1 sort
     * */
    void processNewTasks();

    /**
     * sorts all leftover tasks
     * */
    template <typename CMP>
    void secondPhase(const CMP &Cmp);
};

//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------

template <typename Value>
template <typename CMP>
void QUICKSORT<Value>::sort(const CMP &Cmp)
{
    firstPhase(Cmp);

    int total2ndPhase = host_1stPhaseTasksAmount + host_2ndPhaseTasksAmount;
    if (total2ndPhase > 0)
        secondPhase(Cmp);

    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;

#ifdef CHECK_RESULT_SORT
    if (!is_sorted(arr))
    {
        std::ofstream out("error.txt");
        out << arr << std::endl;
        out << aux << std::endl;
        out << cuda_tasks << std::endl;
        out << cuda_newTasks << std::endl;
        out << cuda_2ndPhaseTasks << std::endl;

        out << cuda_newTasksAmount << std::endl;
        out << cuda_2ndPhaseTasksAmount << std::endl;

        out << iteration << std::endl;
    }
#endif

    return;
}

//---------------------------------------------------------------------------------------------

template <typename Value>
template <typename CMP>
void QUICKSORT<Value>::firstPhase(const CMP &Cmp)
{
    while (host_1stPhaseTasksAmount > 0)
    {
        if (host_1stPhaseTasksAmount >= maxTasks)
            break;

        if (host_2ndPhaseTasksAmount >= maxTasks) //2nd phase occupies enoughs tasks to warrant premature 2nd phase sort
        {
            int tmp = host_1stPhaseTasksAmount;
            host_1stPhaseTasksAmount = 0;
            secondPhase(Cmp);
            cuda_2ndPhaseTasksAmount = host_2ndPhaseTasksAmount = 0;
            host_1stPhaseTasksAmount = tmp;
        }

        //just in case newly created tasks wouldnt fit
        //bite the bullet and sort with single blocks
        if (host_1stPhaseTasksAmount * 2 >= maxTasks + (maxTasks - host_2ndPhaseTasksAmount))
        {
            if (host_2ndPhaseTasksAmount >= 0.75 * maxTasks) //2nd phase occupies enoughs tasks to warrant premature 2nd phase sort
            {
                int tmp = host_1stPhaseTasksAmount;
                host_1stPhaseTasksAmount = 0;
                secondPhase(Cmp);
                cuda_2ndPhaseTasksAmount = host_2ndPhaseTasksAmount = 0;
                host_1stPhaseTasksAmount = tmp;
            }
            else
                break;
        }

        //---------------------------------------------------------------

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
            cudaQuickSort1stPhase<Value, CMP, true>
                <<<blocksCnt, threadsPerBlock, externMemByteSize>>>(
                    arr, aux, Cmp, elemPerBlock,
                    task, cuda_blockToTaskMapping);
        }
        else
        {
            cudaQuickSort1stPhase<Value, CMP, false>
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
template <typename CMP>
void QUICKSORT<Value>::secondPhase(const CMP &Cmp)
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

        cudaQuickSort2ndPhase<Value, CMP, stackSize>
            <<<total2ndPhase, threadsPerBlock, externSharedByteSize>>>(arr, aux, Cmp, tasks, tasks2, elemInShared, desired_2ndPhasElemPerBlock);
    }
    else if (host_1stPhaseTasksAmount > 0)
    {
        auto tasks = leftoverTasks.getView(0, host_1stPhaseTasksAmount);
        cudaQuickSort2ndPhase<Value, CMP, stackSize>
            <<<total2ndPhase, threadsPerBlock, externSharedByteSize>>>(arr, aux, Cmp, tasks, elemInShared, desired_2ndPhasElemPerBlock);
    }
    else
    {
        auto tasks2 = cuda_2ndPhaseTasks.getView(0, host_2ndPhaseTasksAmount);

        cudaQuickSort2ndPhase<Value, CMP, stackSize>
            <<<total2ndPhase, threadsPerBlock, externSharedByteSize>>>(arr, aux, Cmp, tasks2, elemInShared, desired_2ndPhasElemPerBlock);
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
    return Algorithms::reduce<Devices::Cuda>(0, host_1stPhaseTasksAmount, fetch, reduction, 0);
}

template <typename Value>
int QUICKSORT<Value>::getElemPerBlock() const
{
    return desiredElemPerBlock;

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
template <typename CMP>
int QUICKSORT<Value>::initTasks(int elemPerBlock, const CMP &Cmp)
{

    auto &src = iteration % 2 == 0 ? arr : aux;
    auto &tasks = iteration % 2 == 0 ? cuda_tasks : cuda_newTasks;

    //--------------------------------------------------------
    int blocks = host_1stPhaseTasksAmount / threadsPerBlock + (host_1stPhaseTasksAmount % threadsPerBlock != 0);

    cudaCalcBlocksNeeded<<<blocks, threadsPerBlock>>>(tasks.getView(0, host_1stPhaseTasksAmount), elemPerBlock,
                                                      cuda_reductionTaskInitMem.getView(0, host_1stPhaseTasksAmount));
    //cuda_reductionTaskInitMem[i] == how many blocks task i needs
    
    auto reduce = [] __cuda_callable__(const int &a, const int &b) { return a + b; };

    Algorithms::Scan<Devices::Cuda, Algorithms::ScanType::Inclusive >::
        perform(cuda_reductionTaskInitMem, 0, cuda_reductionTaskInitMem.getSize(), reduce, 0);
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

template <typename Value, typename CMP>
void quicksort(ArrayView<Value, Devices::Cuda> arr, const CMP &Cmp)
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

        } // namespace detail
    } // namespace Algorithms
}// namespace TNL