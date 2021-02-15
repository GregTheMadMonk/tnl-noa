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

__global__ void bitonicMergeStep(ArrayView<int, Device> arr,
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

    int s = begin + blockIdx.x * (2 * blockDim.x) + threadIdx.x;
    int e = s + blockDim.x;
    //copy from globalMem into sharedMem
    {
        sharedMem[threadIdx.x] = arr[s];
        if(e < end)
            sharedMem[threadIdx.x + blockDim.x] = arr[e];

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
        for (; len > 1; len /= 2, partsInSeq *= 2)
        {
            __syncthreads();

            {
                int part = i / (len / 2);

                int arrCmpS = begin + part * len + (i % (len / 2));
                int arrCmpE = arrCmpS + len / 2;
                if(arrCmpE >= end)
                    continue;
            }

            int part = threadIdx.x / (len / 2);
            int s = part * len + (threadIdx.x % (len / 2));
            int e = s + len / 2;

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
        arr[s] = sharedMem[threadIdx.x];
        if(e < end)
            arr[e] = sharedMem[threadIdx.x + blockDim.x];
        __syncthreads();
    }
}

//---------------------------------------------

void bitonicSort(ArrayView<int, Device> arr, int begin, int end, bool sortAscending)
{
    int arrSize = end - begin;
    int paddedSize = closestPow2(arrSize);

    int threadsNeeded = arrSize / 2 + (arrSize %2 !=0);

    const int maxThreadsPerBlock = 256;
    int threadPerBlock = min(maxThreadsPerBlock, threadsNeeded);
    int blocks = threadsNeeded / threadPerBlock + (threadsNeeded % threadPerBlock == 0 ? 0 : 1);

    const int sharedMemSize = threadPerBlock * 2 * sizeof(int);

    for (int monotonicSeqLen = 2; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2)
    {
        for (int len = monotonicSeqLen, partsInSeq = 1; len > 1; len /= 2, partsInSeq *= 2)
        {
            if(monotonicSeqLen > sharedMemSize)
            {
                bitonicMergeStep<<<blocks, threadPerBlock>>>(arr, begin, end, sortAscending,
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