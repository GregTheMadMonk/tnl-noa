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

__global__ void bitonicMergeGlobal(ArrayView<int, Device> arr,
                                 int begin, int end, bool sortAscending,
                                 int monotonicSeqLen, int len, int partsInSeq)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int part = i / (len / 2);

    int s = begin + part * len + (i % (len / 2));
    int e = s + len / 2;
    if (e >= end)
        return;

    //calculate the direction of swapping
    int monotonicSeqIdx = part / partsInSeq;
    bool ascending = (monotonicSeqIdx % 2) == 0 ? !sortAscending : sortAscending;

    //special case for parts with no "partner"
    if ((monotonicSeqIdx + 1) * monotonicSeqLen >= end)
        ascending = sortAscending;

    auto &a = arr[s];
    auto &b = arr[e];
    if ((ascending && a > b) || (!ascending && a < b))
        TNL::swap(a, b);
}

__global__ void bitonicMergeSharedMemory(ArrayView<int, Device> arr,
                                         int begin, int end, bool sortAscending,
                                         int monotonicSeqLen, int len, int partsInSeq)
{
    extern __shared__ int sharedMem[];
    int sharedMemLen = 2*blockDim.x;

    int myBlockStart = begin + blockIdx.x * sharedMemLen;
    int myBlockEnd = end < myBlockStart+sharedMemLen? end : myBlockStart+sharedMemLen;

    int copy1 = myBlockStart + threadIdx.x;
    int copy2 = copy1 + blockDim.x;
    //copy from globalMem into sharedMem
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

        //do bitonic sort
        for (int len = monotonicSeqLen, partsInSeq = 1; len > 1; len /= 2, partsInSeq *= 2)
        {
            __syncthreads();

            int part = threadIdx.x / (len / 2);
            int s = part * len + (threadIdx.x % (len / 2));
            int e = s + len / 2;
            if(e >= myBlockEnd - myBlockStart)
                continue;

            //swap
            int a = sharedMem[s], b = sharedMem[e];
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
        __syncthreads();
    }
}

//---------------------------------------------
__global__ void bitoniSort1stStepSharedMemory(ArrayView<int, Device> arr, int begin, int end, bool sortAscending)
{
    extern __shared__ int sharedMem[];
    int sharedMemLen = 2*blockDim.x;

    int myBlockStart = begin + blockIdx.x * sharedMemLen;
    int myBlockEnd = end < myBlockStart+sharedMemLen? end : myBlockStart+sharedMemLen;

    int copy1 = myBlockStart + threadIdx.x;
    int copy2 = copy1 + blockDim.x;
    //copy from globalMem into sharedMem
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

            //special case for parts with no "partner"
            if ((monotonicSeqIdx + 1) * monotonicSeqLen >= end)
                ascending = sortAscending;

            for (int len = monotonicSeqLen, partsInSeq = 1; len > 1; len /= 2, partsInSeq *= 2)
            {
                __syncthreads();

                int part = threadIdx.x / (len / 2);
                int s = part * len + (threadIdx.x % (len / 2));
                int e = s + len / 2;
                if(e >= myBlockEnd - myBlockStart)
                    continue;

                //swap
                int a = sharedMem[s], b = sharedMem[e];
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
        __syncthreads();
    }
}


//---------------------------------------------

void bitonicSort(ArrayView<int, Device> arr, int begin, int end, bool sortAscending)
{
    int arrSize = end - begin;
    int paddedSize = closestPow2(arrSize);

    int threadsNeeded = arrSize / 2 + (arrSize %2 !=0);

    const int maxThreadsPerBlock = 512;
    int threadPerBlock = min(maxThreadsPerBlock, threadsNeeded);
    int blocks = threadsNeeded / threadPerBlock + (threadsNeeded % threadPerBlock == 0 ? 0 : 1);

    const int sharedMemLen = threadPerBlock * 2;
    const int sharedMemSize = sharedMemLen* sizeof(int);

    //---------------------------------------------------------------------------------

    
    bitoniSort1stStepSharedMemory<<<blocks, threadPerBlock, sharedMemSize>>>(arr, begin, end, sortAscending);
    cudaDeviceSynchronize();
    
    for (int monotonicSeqLen = 2*sharedMemLen; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2)
    {
        for (int len = monotonicSeqLen, partsInSeq = 1; len > 1; len /= 2, partsInSeq *= 2)
        {
            if(len > sharedMemLen)
            {
                bitonicMergeGlobal<<<blocks, threadPerBlock>>>(arr, begin, end, sortAscending,
                                                            monotonicSeqLen, len, partsInSeq);
                cudaDeviceSynchronize();
            }
            else
            {

                bitonicMergeSharedMemory<<<blocks, threadPerBlock, sharedMemSize>>>(arr, begin, end, sortAscending,
                                                                                    monotonicSeqLen, len, partsInSeq);
                cudaDeviceSynchronize();
                break;
            }
        }
    }
}

//---------------------------------------------

void bitonicSort(ArrayView<int, Device> arr, bool sortAscending = true)
{
    bitonicSort(arr, 0, arr.getSize(), sortAscending);
}
