#include <TNL/Containers/Array.h>
#include <iostream>

using namespace TNL;
using namespace TNL::Containers;

typedef Devices::Cuda Device;

#define deb(x) std::cout << #x << " = " << x << std::endl;

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

//---------------------------------------------
/**
 * this kernel simulates 1 exchange 
 */
template <typename Value>
__global__ void bitonicMergeGlobal(ArrayView<Value, Device> arr,
                                 int begin, int end, bool sortAscending,
                                 int monotonicSeqLen, int len, int partsInSeq)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int part = i / (len / 2); //computes which sorting block this thread belongs to

    //the index of 2 elements that should be compared and swapped
    int s = begin + part * len + (i % (len / 2));
    int e = s + len / 2;
    if (e >= end) //arr[e] is virtual padding and will not be exchanged with
        return;

    //calculate the direction of swapping
    int monotonicSeqIdx = part / partsInSeq;
    bool ascending = (monotonicSeqIdx % 2) == 0 ? !sortAscending : sortAscending;
    if ((monotonicSeqIdx + 1) * monotonicSeqLen >= end) //special case for part with no "partner" to be merged with in next phase
        ascending = sortAscending;

    //cmp and swap
    auto &a = arr[s];
    auto &b = arr[e];
    if ((ascending && a > b) || (!ascending && a < b))
        TNL::swap(a, b);
}

//---------------------------------------------
/**
 * kernel for merging if whole block fits into shared memory
 * will merge all the way down til stride == 2
 * */
template <typename Value>
__global__ void bitonicMergeSharedMemory(ArrayView<Value, Device> arr,
                                         int begin, int end, bool sortAscending,
                                         int monotonicSeqLen, int len, int partsInSeq)
{
    extern __shared__ Value sharedMem[];
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

        bool ascending = (monotonicSeqIdx % 2) == 0 ? !sortAscending : sortAscending;
        //special case for parts with no "partner"
        if ((monotonicSeqIdx + 1) * monotonicSeqLen >= end)
            ascending = sortAscending;
        //------------------------------------------

        //do bitonic merge
        for (int len = monotonicSeqLen, partsInSeq = 1; len > 1; len /= 2, partsInSeq *= 2)
        {
            __syncthreads();

            //calculates which 2 indexes will be compared and swap
            int part = threadIdx.x / (len / 2);
            int s = part * len + (threadIdx.x % (len / 2));
            int e = s + len / 2;
            if(e >= myBlockEnd - myBlockStart) //touching virtual padding
                continue;

            //cmp and swap
            Value a = sharedMem[s], b = sharedMem[e];
            if ((ascending && a > b) || (!ascending && a < b))
            {
                sharedMem[s] = b;
                sharedMem[e] = a;
            }
        }

        __syncthreads();

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
/**
 * very similar to bitonicMergeSharedMemory
 * does bitonicMergeSharedMemory but afterwards increases monotoncSeqLen
 *  then trickles down again
 * this continues until whole sharedMem is sorted
 * */
template <typename Value>
__global__ void bitoniSort1stStepSharedMemory(ArrayView<Value, Device> arr, int begin, int end, bool sortAscending)
{
    extern __shared__ Value sharedMem[];
    int sharedMemLen = 2*blockDim.x;

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
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int paddedSize = closestPow2(myBlockEnd - myBlockStart);

        for (int monotonicSeqLen = 2; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2)
        {
            //calculate the direction of swapping
            int monotonicSeqIdx = i / (monotonicSeqLen/2);
            bool ascending = (monotonicSeqIdx % 2) == 0 ? !sortAscending : sortAscending;
            if ((monotonicSeqIdx + 1) * monotonicSeqLen >= end) //special case for parts with no "partner"
                ascending = sortAscending;

            for (int len = monotonicSeqLen, partsInSeq = 1; len > 1; len /= 2, partsInSeq *= 2)
            {
                __syncthreads();

                //calculates which 2 indexes will be compared and swap
                int part = threadIdx.x / (len / 2);
                int s = part * len + (threadIdx.x % (len / 2));
                int e = s + len / 2;
                if(e >= myBlockEnd - myBlockStart) //touching virtual padding
                    continue;

                //cmp and swap
                Value a = sharedMem[s], b = sharedMem[e];
                if ((ascending && a > b) || (!ascending && a < b))
                {
                    sharedMem[s] = b;
                    sharedMem[e] = a;
                }
            }
        }

        __syncthreads();

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
template <typename Value>
void bitonicSort(ArrayView<Value, Device> arr, int begin, int end, bool sortAscending)
{
    int arrSize = end - begin;
    int paddedSize = closestPow2(arrSize);

    int threadsNeeded = arrSize / 2 + (arrSize %2 !=0);

    const int maxThreadsPerBlock = 512;
    int threadPerBlock = min(maxThreadsPerBlock, threadsNeeded);
    int blocks = threadsNeeded / threadPerBlock + (threadsNeeded % threadPerBlock == 0 ? 0 : 1);

    const int sharedMemLen = threadPerBlock * 2;
    const int sharedMemSize = sharedMemLen* sizeof(Value);

    //---------------------------------------------------------------------------------

    
    bitoniSort1stStepSharedMemory<<<blocks, threadPerBlock, sharedMemSize>>>(arr, begin, end, sortAscending);
    
    for (int monotonicSeqLen = 2*sharedMemLen; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2)
    {
        for (int len = monotonicSeqLen, partsInSeq = 1; len > 1; len /= 2, partsInSeq *= 2)
        {
            if(len > sharedMemLen)
            {
                bitonicMergeGlobal<<<blocks, threadPerBlock>>>(arr, begin, end, sortAscending,
                                                            monotonicSeqLen, len, partsInSeq);
            }
            else
            {

                bitonicMergeSharedMemory<<<blocks, threadPerBlock, sharedMemSize>>>(arr, begin, end, sortAscending,
                                                                                    monotonicSeqLen, len, partsInSeq);
                break;
            }
        }
    }
    cudaDeviceSynchronize();
}

//---------------------------------------------
template <typename Value>
void bitonicSort(ArrayView<Value, Device> arr, bool sortAscending = true)
{
    bitonicSort(arr, 0, arr.getSize(), sortAscending);
}
