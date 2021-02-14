#include <TNL/Containers/Array.h>

using namespace TNL;
using namespace TNL::Containers;

typedef Devices::Cuda Device;

//---------------------------------------------

__host__ __device__ int closestPow2(int x)
{
    if(x ==0)
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
    
    int arrSize = end - begin;
    int paddedSize = closestPow2(arrSize);
    //------------------------------------------

    //copy from globalMem into sharedMem
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int part = i / (len / 2);

        int s = begin + part * len + threadIdx.x;
        int e = s + blockDim.x/2;

        sharedMem[threadIdx.x] = arr[s];
        sharedMem[threadIdx.x + blockDim.x/2] = e < end? arr[e] : -1; //any default value is ok
        __syncthreads();
    }
    
    //------------------------------------------

    //do bitonic sort
    for (; len > 1; len /= 2, partsInSeq *= 2)
    {
        __syncthreads();

        int i = threadIdx.x;

        int s = part * len + (i% (len / 2));
        int e = s + len / 2;


        //calculate the direction of swapping
        int monotonicSeqIdx = part / partsInSeq;
        bool ascending = (monotonicSeqIdx % 2) == 0 ? !sortAscending : sortAscending;

        //special case for parts with no "partner"
        if ((monotonicSeqIdx + 1) * monotonicSeqLen >= end)
            ascending = sortAscending;

        //swap
        /*
        if ((ascending && a > b) || (!ascending && a < b))
        {
            //TNL::swap(a, b);
        }
        */
    }

    //------------------------------------------
    //writeback to global memory
    
}

//---------------------------------------------

void bitonicSort(ArrayView<int, Device> arr, int begin, int end, bool sortAscending)
{
    int arrSize = end - begin;
    int paddedSize = closestPow2(arrSize);

    int threadPerBlock = 256;
    int blocks = arrSize/threadPerBlock + (arrSize%threadPerBlock == 0? 0 : 1);

    for (int monotonicSeqLen = 2; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2)
    {
        for (int len = monotonicSeqLen, partsInSeq = 1; len > 1; len /= 2, partsInSeq *= 2)
        {
            bitonicMergeStep<<<blocks, threadPerBlock>>>(arr, begin, end, sortAscending, 
                                                        monotonicSeqLen, len, partsInSeq);
        }
    }
}

//---------------------------------------------

void bitonicSort(ArrayView<int, Device> arr, bool sortAscending = true)
{
    bitonicSort(arr, 0, arr.getSize(), sortAscending);
}