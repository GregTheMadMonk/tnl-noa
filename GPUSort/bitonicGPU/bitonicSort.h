#pragma once
#include <TNL/Containers/Array.h>

//---------------------------------------------

__host__ __device__ int closestPow2(int x)
{
    if (x == 0)
        return 0;

    int ret = 1;
    while (ret < x)
        ret <<= 1;

    return ret;
}

template <typename Value, typename Function>
__host__ __device__ void cmpSwap(Value & a, Value &b, bool ascending, const Function & Cmp)
{
    if( (ascending == Cmp(b, a)))
        TNL::swap(a, b);
}
//---------------------------------------------
/**
 * this kernel simulates 1 exchange 
 */
template <typename Value, typename Function>
__global__ void bitonicMergeGlobal(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr,
                                 int begin, int end, const Function & Cmp,
                                 int monotonicSeqLen, int len, int partsInSeq)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int part = i / (len / 2); //computes which sorting block this thread belongs to

    //the index of 2 elements that should be compared and swapped
    int s = begin + part * len + (i & ((len / 2) - 1) );
    int e = s + len / 2;
    if (e >= end) //arr[e] is virtual padding and will not be exchanged with
        return;

    //calculate the direction of swapping
    int monotonicSeqIdx = part / partsInSeq;
    bool ascending = (monotonicSeqIdx & 1) != 0;
    if ((monotonicSeqIdx + 1) * monotonicSeqLen >= end) //special case for part with no "partner" to be merged with in next phase
        ascending = true;

    cmpSwap(arr[s], arr[e], ascending, Cmp);
}

//---------------------------------------------
/**
 * kernel for merging if whole block fits into shared memory
 * will merge all the way down til stride == 2
 * */
template <typename Value, typename Function>
__global__ void bitonicMergeSharedMemory(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr,
                                         int begin, int end, const Function & Cmp,
                                         int monotonicSeqLen, int len, int partsInSeq)
{
    extern __shared__ int externMem[];
    Value * sharedMem = (Value *)externMem;

    int sharedMemLen = 2*blockDim.x;

    //1st index and last index of subarray that this threadBlock should merge
    int myBlockStart = begin + blockIdx.x * sharedMemLen;
    int myBlockEnd = end < myBlockStart+sharedMemLen? end : myBlockStart+sharedMemLen;

    //copy from globalMem into sharedMem
    int copy1 = myBlockStart + threadIdx.x;
    int copy2 = copy1 + blockDim.x;
    {
        if(copy1 < end)
            sharedMem[threadIdx.x] = arr[copy1];
        if(copy2 < end)
            sharedMem[threadIdx.x + blockDim.x] = arr[copy2];

        __syncthreads();
    }
    
    //------------------------------------------
    //bitonic activity
    {
        //calculate the direction of swapping
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int part = i / (len / 2);
        int monotonicSeqIdx = part / partsInSeq;

        bool ascending = (monotonicSeqIdx & 1) != 0;
        //special case for parts with no "partner"
        if ((monotonicSeqIdx + 1) * monotonicSeqLen >= end)
            ascending = true;
        //------------------------------------------

        //do bitonic merge
        for (; len > 1; len /= 2)
        {
            //calculates which 2 indexes will be compared and swap
            int part = threadIdx.x / (len / 2);
            int s = part * len + (threadIdx.x & ((len /2) - 1));
            int e = s + len / 2;

            if(e < myBlockEnd - myBlockStart) //touching virtual padding
                cmpSwap(sharedMem[s], sharedMem[e], ascending, Cmp);
            __syncthreads();
        }
    }

    //------------------------------------------
    
    //writeback to global memory
    {
        if(copy1 < end)
            arr[copy1] = sharedMem[threadIdx.x];
        if(copy2 < end)
            arr[copy2] = sharedMem[threadIdx.x + blockDim.x];
    }
}

//---------------------------------------------

template <typename Value, typename Function>
__device__ void bitonicSort_Block(
                TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr,
                int myBlockStart, int myBlockEnd, Value* sharedMem, const Function & Cmp)
{

    //copy from globalMem into sharedMem
    int copy1 = myBlockStart + threadIdx.x;
    int copy2 = copy1 + blockDim.x;
    {
        if(copy1 < myBlockEnd)
            sharedMem[threadIdx.x] = arr[copy1];

        if(copy2 < myBlockEnd)
            sharedMem[threadIdx.x + blockDim.x] = arr[copy2];

        __syncthreads();
    }
    
    //------------------------------------------
    //bitonic activity
    {
        int i = threadIdx.x;
        int paddedSize = closestPow2(myBlockEnd - myBlockStart);

        for (int monotonicSeqLen = 2; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2)
        {
            //calculate the direction of swapping
            int monotonicSeqIdx = i / (monotonicSeqLen/2);
            bool ascending = (monotonicSeqIdx & 1) != 0;
            if ((monotonicSeqIdx + 1) * monotonicSeqLen >= myBlockEnd) //special case for parts with no "partner"
                ascending = true;

            for (int len = monotonicSeqLen; len > 1; len /= 2)
            {
                //calculates which 2 indexes will be compared and swap
                int part = threadIdx.x / (len / 2);
                int s = part * len + (threadIdx.x & ((len / 2) - 1));
                int e = s + len / 2;

                if(e < myBlockEnd - myBlockStart) //touching virtual padding
                    cmpSwap(sharedMem[s], sharedMem[e], ascending, Cmp);
                __syncthreads();
            }
        }
    }

    //------------------------------------------
    //writeback to global memory
    {
        if(copy1 < myBlockEnd)
            arr[copy1] = sharedMem[threadIdx.x];
        if(copy2 < myBlockEnd)
            arr[copy2] = sharedMem[threadIdx.x + blockDim.x];
    }
}
/**
 * very similar to bitonicMergeSharedMemory
 * does bitonicMergeSharedMemory but afterwards increases monotoncSeqLen
 *  then trickles down again
 * this continues until whole sharedMem is sorted
 * */
template <typename Value, typename Function>
__global__ void bitoniSort1stStepSharedMemory(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr,
                                            int begin, int end, const Function & Cmp)
{
    extern __shared__ int externMem[];
    int sharedMemLen = 2*blockDim.x;
    int myBlockStart = begin + blockIdx.x * sharedMemLen;
    int myBlockEnd = end < myBlockStart+sharedMemLen? end : myBlockStart+sharedMemLen;

    if(blockIdx.x%2 == 0)
        bitonicSort_Block(arr, myBlockStart, myBlockEnd, (Value*) externMem, Cmp);
    else
        bitonicSort_Block(arr, myBlockStart, myBlockEnd, (Value*) externMem, 
            [&] __cuda_callable__ (const Value&a, const Value&b){return Cmp(b, a);}
    );
}


//---------------------------------------------
template <typename Value, typename Function>
void bitonicSort(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr, int begin, int end, const Function& Cmp)
{
    int arrSize = end - begin;
    int paddedSize = closestPow2(arrSize);

    int threadsNeeded = arrSize / 2 + (arrSize %2 !=0);

    const int maxThreadsPerBlock = 512;
    int threadPerBlock = maxThreadsPerBlock;
    int blocks = threadsNeeded / threadPerBlock + (threadsNeeded % threadPerBlock == 0 ? 0 : 1);

    const int sharedMemLen = threadPerBlock * 2;
    const int sharedMemSize = sharedMemLen* sizeof(Value);

    //---------------------------------------------------------------------------------

    
    bitoniSort1stStepSharedMemory<<<blocks, threadPerBlock, sharedMemSize>>>(arr, begin, end, Cmp);
    
    for (int monotonicSeqLen = 2*sharedMemLen; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2)
    {
        for (int len = monotonicSeqLen, partsInSeq = 1; len > 1; len /= 2, partsInSeq *= 2)
        {
            if(len > sharedMemLen)
            {
                bitonicMergeGlobal<<<blocks, threadPerBlock>>>(arr, begin, end, Cmp,
                                                            monotonicSeqLen, len, partsInSeq);
            }
            else
            {

                bitonicMergeSharedMemory<<<blocks, threadPerBlock, sharedMemSize>>>(arr, begin, end, Cmp,
                                                                                    monotonicSeqLen, len, partsInSeq);
                break;
            }
        }
    }
    cudaDeviceSynchronize();
}

//---------------------------------------------

template <typename Value, typename Function>
void bitonicSort(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr, int begin, int end)
{
    bitonicSort(arr, begin, end, [] __cuda_callable__ (const Value & a, const Value & b) {return a < b;});
}

template <typename Value, typename Function>
void bitonicSort(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr, const Function & Cmp)
{
    bitonicSort(arr, 0, arr.getSize(), Cmp);
}

template <typename Value>
void bitonicSort(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr)
{
    bitonicSort(arr, [] __cuda_callable__ (const Value & a, const Value & b) {return a < b;});
}

//---------------------------------------------
template <typename Value, typename Function>
void bitonicSort(std::vector<Value> & vec, int begin, int end, const Function & Cmp)
{
    TNL::Containers::Array<Value, TNL::Devices::Cuda> Arr(vec);
    auto view = Arr.getView();
    bitonicSort(view, begin, end, Cmp);

    TNL::Algorithms::MultiDeviceMemoryOperations<TNL::Devices::Host, TNL::Devices::Cuda >::
    copy(vec.data(), view.getData(), view.getSize());
}

template <typename Value>
void bitonicSort(std::vector<Value> & vec, int begin, int end)
{
    bitonicSort(vec, begin, end, [] __cuda_callable__ (const Value & a, const Value & b) {return a < b;});
}

template <typename Value, typename Function>
void bitonicSort(std::vector<Value> & vec, const Function & Cmp)
{
    bitonicSort(vec, 0, vec.size(), Cmp);
}

template <typename Value>
void bitonicSort(std::vector<Value> & vec)
{
    bitonicSort(vec, [] __cuda_callable__ (const Value & a, const Value & b) {return a < b;});
}

//---------------------------------------------