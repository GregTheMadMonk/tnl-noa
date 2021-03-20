#pragma once

#include <TNL/Containers/Array.h>
#include "reduction.cuh"
#include "task.h"
#include "../bitonicSort/bitonicSort.h"
#include <iostream>

#define deb(x) std::cout << #x << " = " << x << std::endl;

using CudaArrayView = TNL::Containers::ArrayView<int, TNL::Devices::Cuda>;
using CudaTaskArray = TNL::Containers::Array<TASK, TNL::Devices::Cuda>;

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

template <typename Function>
__global__ void cudaPartition(CudaArrayView arr,const Function & Cmp,
                                CudaArrayView aux,
                              TNL::Containers::ArrayView<int, TNL::Devices::Cuda> cuda_blockToTaskMapping,
                              int elemPerBlock,
                              TNL::Containers::ArrayView<TASK, TNL::Devices::Cuda> cuda_tasks,
                              TNL::Containers::ArrayView<TASK, TNL::Devices::Cuda> cuda_newTasks,
                              int *newTasksCnt,
                              TNL::Containers::ArrayView<TASK, TNL::Devices::Cuda> cuda_2ndPhaseTasks,
                            int * cuda_2ndPhaseCnt
)
{
    static __shared__ TASK myTask;
    static __shared__ int smallerStart, biggerStart;
    static __shared__ int pivot;
    static __shared__ int myTaskIdx;
    static __shared__ bool writePivot;

    if (threadIdx.x == 0)
    {
        myTaskIdx = cuda_blockToTaskMapping[blockIdx.x];
        myTask = cuda_tasks[myTaskIdx];
        pivot = arr[myTask.arrEnd - 1];
        writePivot = false;
    }
    __syncthreads();

    //only works if consecutive blocks work on the same task
    const int myBegin = myTask.arrBegin + elemPerBlock * (blockIdx.x - myTask.firstBlock);
    const int myEnd = TNL::min(myTask.arrEnd, myBegin + elemPerBlock);

    //-------------------------------------------------------------------------

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

    //-----------------------------------------------------------

    if (threadIdx.x == 0 && atomicAdd(&(cuda_tasks[myTaskIdx].stillWorkingCnt), -1) == 1)
    {
        writePivot = true;
        myTask = cuda_tasks[myTaskIdx]; //update auxBeginIdx, auxEndIdx value
    }
    __syncthreads();

    if (!writePivot)
        return;

    for (int i = myTask.auxBeginIdx + threadIdx.x; i < myTask.auxEndIdx; i += blockDim.x)
        aux[i] = pivot;

    if (threadIdx.x != 0)
        return;
        
    if (myTask.auxBeginIdx - myTask.arrBegin > 0) //smaller
    {
        if(myTask.auxBeginIdx - myTask.arrBegin <= blockDim.x*2)
        {
            int newTaskIdx = atomicAdd(cuda_2ndPhaseCnt, 1);
            cuda_2ndPhaseTasks[newTaskIdx] = TASK(
                myTask.arrBegin, myTask.auxBeginIdx,
                myTask.arrBegin, myTask.auxBeginIdx
            );
        }
        else
        {
            int newTaskIdx = atomicAdd(newTasksCnt, 1);
            cuda_newTasks[newTaskIdx] = TASK(
                myTask.arrBegin, myTask.auxBeginIdx,
                myTask.arrBegin, myTask.auxBeginIdx
            );
        }
    }

    if (myTask.arrEnd - myTask.auxEndIdx > 0) //greater
    {
        if (myTask.arrEnd - myTask.auxEndIdx <= blockDim.x*2)
        {
            int newTaskIdx = atomicAdd(cuda_2ndPhaseCnt, 1);
            cuda_2ndPhaseTasks[newTaskIdx] = TASK(
                myTask.auxEndIdx, myTask.arrEnd,
                myTask.auxEndIdx, myTask.arrEnd
            );
        }
        else
        {
            int newTaskIdx = atomicAdd(newTasksCnt, 1);
            cuda_newTasks[newTaskIdx] = TASK(
                myTask.auxEndIdx, myTask.arrEnd,
                myTask.auxEndIdx, myTask.arrEnd
            );
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

    if (i < taskAmount)
    {
        auto task = cuda_tasks[i];
        int size = task.arrEnd - task.arrBegin;
        blocksNeeded = size / elemPerBlock + (size % elemPerBlock != 0);
    }

    int blocksNeeded_total = blockInclusivePrefixSum(blocksNeeded);
    if (threadIdx.x == blockDim.x - 1)
        avail = atomicAdd(firstAvailBlock, blocksNeeded_total);
    __syncthreads();

    if (i < taskAmount)
    {
        int myFirstAvailBlock = avail + blocksNeeded_total - blocksNeeded;

        cuda_tasks[i].firstBlock = myFirstAvailBlock;
        cuda_tasks[i].setBlocks(blocksNeeded);

        for (int set = 0; set < blocksNeeded; set++)
            cuda_blockToTaskMapping[myFirstAvailBlock++] = i;
    }
}


//-----------------------------------------------------------
template<typename Function>
__device__ void cudaQuickSort_block(CudaArrayView arr, const Function & Cmp,
                            CudaArrayView aux,
                            int * stackArrBegin, int *stackArrEnd, int stackSize,
                            int * bitonicMem )
{
    static __shared__ int begin, end;
    static __shared__ int stackTop;
    static __shared__ int pivotBegin, pivotEnd;
    static __shared__ int pivot;

    if (threadIdx.x == 0)
    {
        stackArrBegin[0] = 0;
        stackArrEnd[0] = arr.getSize();
        stackTop = 1;
    }
    __syncthreads();

    while(stackTop > 0)
    {
        if (threadIdx.x == 0)
        {
            begin = stackArrBegin[stackTop - 1];
            end = stackArrEnd[stackTop - 1];
            stackTop--;
            pivot = arr[end - 1];
        }
        __syncthreads();
        
        int size = end - begin;
        if(size<= blockDim.x*2)
        {
            bitonicSort_Block(arr.getView(begin, end), arr.getView(begin, end), bitonicMem, Cmp);
            continue;
        }

        int smaller = 0, bigger = 0;
        cmpElem(arr, begin, end, smaller, bigger, pivot);

        int smallerOffset = blockInclusivePrefixSum(smaller);
        int biggerOffset = blockInclusivePrefixSum(bigger);

        if (threadIdx.x == blockDim.x - 1)
        {
            pivotBegin = begin + smallerOffset;
            pivotEnd = end - biggerOffset;
        }
        __syncthreads();

        int destSmaller = smallerOffset - smaller;
        int destBigger = pivotEnd  + biggerOffset - bigger;
        copyData(arr, begin, end, aux, destSmaller, destBigger, pivot);
        __syncthreads();

        for (int i = begin + threadIdx.x; i < end; i += blockDim.x)
        {
            if(i >= pivotBegin && i < pivotEnd)
                arr[i] = pivot;
            else
                arr[i] = aux[i];
        }

        if(threadIdx.x == 0)
        {
            if(pivotBegin - begin > 1) //left from pivot are smaller elems
            {
                stackArrBegin[stackTop] = begin;
                stackArrEnd[stackTop] = pivotBegin;
                stackTop++;
            }
            
            if(end - pivotEnd > 1) //right from pivot until end are elem greater than pivot
            {
                stackArrBegin[stackTop] = pivotEnd;
                stackArrEnd[stackTop] = end;
                stackTop++;
            }
        }
        __syncthreads();
    }
}

template<typename Function>
__global__
void cudaQuickSort(CudaArrayView arr, const Function & Cmp,
                    CudaArrayView aux, int stackSize,
                    TNL::Containers::ArrayView<TASK, TNL::Devices::Cuda> cuda_tasks)
{
    extern __shared__ int externMem[];

    static __shared__ TASK task;
    if(threadIdx.x == 0)
        task = cuda_tasks[blockIdx.x];
    __syncthreads();

    int * bitonicMem = externMem;
    int * stackLeft = bitonicMem + (2*blockDim.x);
    int * stackRight = stackLeft+ (stackSize/2);
    
    cudaQuickSort_block(arr.getView(task.arrBegin, task.arrEnd), Cmp, 
        aux.getView(task.auxBeginIdx, task.auxEndIdx),
        stackLeft, stackRight, stackSize/2,
        bitonicMem
    );
    
}

//-----------------------------------------------------------
//-----------------------------------------------------------
const int threadsPerBlock = 512, maxBlocks = 1 << 15; //32k
const int maxTasks = 1<<10;
const int minElemPerBlock = threadsPerBlock*2;

class QUICKSORT
{
    CudaArrayView arr;
    TNL::Containers::Array<int, TNL::Devices::Cuda> aux;

    CudaTaskArray cuda_tasks, cuda_newTasks, cuda_2ndPhaseTasks;

    TNL::Containers::Array<int, TNL::Devices::Cuda> cuda_newTasksAmount, cuda_2ndPhaseTasksAmount; //is in reality 1 integer
    int tasksAmount; //counter for Host == cuda_newTasksAmount
    int totalTask; // cuda_newTasksAmount + cuda_2ndPhaseTasksAmount

    TNL::Containers::Array<int, TNL::Devices::Cuda> cuda_blockToTaskMapping;
    TNL::Containers::Array<int, TNL::Devices::Cuda> cuda_blockToTaskMapping_Cnt; //is in reality 1 integer

    int iteration = 0;

    //--------------------------------------------------------------------------------------
public:
    QUICKSORT(CudaArrayView _arr)
        : arr(_arr), aux(arr.getSize()),
          cuda_tasks(maxBlocks), cuda_newTasks(maxBlocks), cuda_2ndPhaseTasks(maxBlocks),
          cuda_newTasksAmount(1), cuda_2ndPhaseTasksAmount(1),
          cuda_blockToTaskMapping(maxBlocks), cuda_blockToTaskMapping_Cnt(1)
    {
        cuda_tasks.setElement(0, TASK(0, arr.getSize(), 0, arr.getSize()));
        totalTask = tasksAmount = 1;
        cuda_2ndPhaseTasksAmount = 0;
    }

    template<typename Function>
    void sort(const Function & Cmp)
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
                    cuda_2ndPhaseTasks.getView(), cuda_2ndPhaseTasksAmount.getData()
                );
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
                    cuda_2ndPhaseTasks.getView(), cuda_2ndPhaseTasksAmount.getData()
                );
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

    template<typename Function>
    void _2ndPhase(const Function & Cmp)
    {
        if(totalTask == 0) return;
        
        TNL::Algorithms::MultiDeviceMemoryOperations<TNL::Devices::Cuda, TNL::Devices::Cuda>::
        copy(cuda_2ndPhaseTasks.getData() + (totalTask - tasksAmount),
            (iteration%2? cuda_newTasks.getData() :cuda_tasks.getData() ),
            tasksAmount
        );
        
        int blocks = totalTask;

        int stackSize = 128, stackMem = stackSize * sizeof(int); 
        int bitonicMem = threadsPerBlock*2*sizeof(int);
        int auxMem = stackMem + bitonicMem;
        cudaQuickSort<<<blocks, threadsPerBlock, auxMem>>>(arr, Cmp, aux.getView(), stackSize, cuda_2ndPhaseTasks.getView());
    }
};

//-----------------------------------------------------------

template<typename Function>
void quicksort(CudaArrayView arr, const Function & Cmp)
{
    QUICKSORT sorter(arr);
    sorter.sort(Cmp);
}

void quicksort(CudaArrayView arr)
{
    quicksort(arr, []__cuda_callable__(int a, int b){return a < b;});
}
