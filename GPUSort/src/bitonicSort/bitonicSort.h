#pragma once
#include <TNL/Containers/Array.h>

//---------------------------------------------

// Inline PTX call to return index of highest non-zero bit in a word
static __device__ __forceinline__ unsigned int __btflo(unsigned int word)
{
    unsigned int ret;
    asm volatile("bfind.u32 %0, %1;" : "=r"(ret) : "r"(word));
    return ret;
}

__device__ int closestPow2_ptx(int len)
{
    return 1 << (__btflo((unsigned)len-1U)+1);
}

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
__host__ __device__ void cmpSwap(Value &a, Value &b, bool ascending, const Function &Cmp)
{
    if (ascending == Cmp(b, a))
        TNL::swap(a, b);
}

//---------------------------------------------

/**
 * this kernel simulates 1 exchange 
 * splits input arr that is bitonic into 2 bitonic sequences
 */
template <typename Value, typename Function>
__global__ void bitonicMergeGlobal(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr,
                                   const Function &Cmp,
                                   int monotonicSeqLen, int len, int partsInSeq)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int part = i / (len / 2); //computes which sorting block this thread belongs to

    //the index of 2 elements that should be compared and swapped
    int s = part * len + (i & ((len / 2) - 1));
    int e = s + len / 2;
    if (e >= arr.getSize()) //arr[e] is virtual padding and will not be exchanged with
        return;

    //calculate the direction of swapping
    int monotonicSeqIdx = part / partsInSeq;
    bool ascending = (monotonicSeqIdx & 1) != 0;
    if ((monotonicSeqIdx + 1) * monotonicSeqLen >= arr.getSize()) //special case for part with no "partner" to be merged with in next phase
        ascending = true;

    cmpSwap(arr[s], arr[e], ascending, Cmp);
}
//---------------------------------------------
//---------------------------------------------

/**
 * simulates many layers of merge
 * turns input that is a bitonic sequence into 1 monotonic sequence
 * 
 * this version uses shared memory to do the operations
 * */
template <typename Value, typename Function>
__global__ void bitonicMergeSharedMemory(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr,
                                         const Function &Cmp,
                                         int monotonicSeqLen, int len, int partsInSeq)
{
    extern __shared__ int externMem[];
    Value *sharedMem = (Value *)externMem;

    int sharedMemLen = 2 * blockDim.x;

    //1st index and last index of subarray that this threadBlock should merge
    int myBlockStart = blockIdx.x * sharedMemLen;
    int myBlockEnd = TNL::min(arr.getSize(), myBlockStart + sharedMemLen);

    //copy from globalMem into sharedMem
    for(int i = threadIdx.x; myBlockStart + i < myBlockEnd; i += blockDim.x)
        sharedMem[i] = arr[myBlockStart + i];
    __syncthreads();

    //------------------------------------------
    //bitonic activity
    {
        //calculate the direction of swapping
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int part = i / (len / 2);
        int monotonicSeqIdx = part / partsInSeq;

        bool ascending = (monotonicSeqIdx & 1) != 0;
        //special case for parts with no "partner"
        if ((monotonicSeqIdx + 1) * monotonicSeqLen >= arr.getSize())
            ascending = true;
        //------------------------------------------

        //do bitonic merge
        for (; len > 1; len /= 2)
        {
            //calculates which 2 indexes will be compared and swap
            int part = threadIdx.x / (len / 2);
            int s = part * len + (threadIdx.x & ((len / 2) - 1));
            int e = s + len / 2;

            if (e < myBlockEnd - myBlockStart) //not touching virtual padding
                cmpSwap(sharedMem[s], sharedMem[e], ascending, Cmp);
            __syncthreads();
        }
    }

    //------------------------------------------

    //writeback to global memory
    for(int i = threadIdx.x; myBlockStart + i < myBlockEnd; i += blockDim.x)
        arr[myBlockStart + i] = sharedMem[i];
}

/**
 * simulates many layers of merge
 * turns input that is a bitonic sequence into 1 monotonic sequence
 * 
 * this user only operates on global memory, no shared memory is used
 * */
template <typename Value, typename Function>
__global__ void bitonicMerge(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr,
                             const Function &Cmp,
                             int monotonicSeqLen, int len, int partsInSeq)
{
    //1st index and last index of subarray that this threadBlock should merge
    int myBlockStart = blockIdx.x * (2 * blockDim.x);
    int myBlockEnd = TNL::min(arr.getSize(), myBlockStart + (2 * blockDim.x));

    auto src = arr.getView(myBlockStart, myBlockEnd);

    //calculate the direction of swapping
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int part = i / (len / 2);
    int monotonicSeqIdx = part / partsInSeq;

    bool ascending = (monotonicSeqIdx & 1) != 0;
    //special case for parts with no "partner"
    if ((monotonicSeqIdx + 1) * monotonicSeqLen >= arr.getSize())
        ascending = true;
    //------------------------------------------

    //do bitonic merge
    for (; len > 1; len /= 2)
    {
        //calculates which 2 indexes will be compared and swap
        int part = threadIdx.x / (len / 2);
        int s = part * len + (threadIdx.x & ((len / 2) - 1));
        int e = s + len / 2;

        if (e < myBlockEnd - myBlockStart) //not touching virtual padding
            cmpSwap(src[s], src[e], ascending, Cmp);
        __syncthreads();
    }
}

//---------------------------------------------

/**
 * IMPORTANT: all threads in block have to call this function to work properly
 * the size of src isn't limited, but for optimal efficiency, no more than 8*blockDim.x should be used
 * Description: sorts src and writes into dst within a block
 * works independently from other concurrent blocks
 * @param sharedMem sharedMem pointer has to be able to store all of src elements
 * */
template <typename Value, typename Function>
__device__ void bitonicSort_Block(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> src,
                                  TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> dst,
                                  Value *sharedMem, const Function &Cmp)
{
    //copy from globalMem into sharedMem
    for(int i = threadIdx.x; i < src.getSize(); i += blockDim.x)
        sharedMem[i] = src[i];
    __syncthreads();

    //------------------------------------------
    //bitonic activity
    {
        int paddedSize = closestPow2_ptx(src.getSize());

        for (int monotonicSeqLen = 2; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2)
        {
            for (int len = monotonicSeqLen; len > 1; len /= 2)
            {
                for(int i = threadIdx.x; ; i+=blockDim.x) //simulates other blocks in case src.size > blockDim.x*2
                {
                    //calculates which 2 indexes will be compared and swap
                    int part = i / (len / 2);
                    int s = part * len + (i & ((len / 2) - 1));
                    int e = s + len / 2;

                    if(e >= src.getSize()) //touching virtual padding, the order dont swap
                        break;

                    //calculate the direction of swapping
                    int monotonicSeqIdx = i / (monotonicSeqLen / 2);
                    bool ascending = (monotonicSeqIdx & 1) != 0;
                    if ((monotonicSeqIdx + 1) * monotonicSeqLen >= src.getSize()) //special case for parts with no "partner"
                        ascending = true;

                    cmpSwap(sharedMem[s], sharedMem[e], ascending, Cmp);
                }
                
                __syncthreads(); //only 1 synchronization needed
            }
        }
    }

    //------------------------------------------
    //writeback to global memory
    for(int i = threadIdx.x; i < dst.getSize(); i += blockDim.x)
        dst[i] = sharedMem[i];
}


/**
 * IMPORTANT: all threads in block have to call this function to work properly
 * IMPORTANT: unlike the counterpart with shared memory, this function only works in-place
 * the size of src isn't limited, but for optimal efficiency, no more than 8*blockDim.x should be used
 * Description: sorts src in place using bitonic sort
 * works independently from other concurrent blocks
 * this version doesnt use shared memory and is prefered for Value with big size
 * */
template <typename Value, typename Function>
__device__ void bitonicSort_Block(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> src,
                                  const Function &Cmp)
{
    int paddedSize = closestPow2_ptx(src.getSize());

    for (int monotonicSeqLen = 2; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2)
    {
        for (int len = monotonicSeqLen; len > 1; len /= 2)
        {
            for(int i = threadIdx.x; ; i+=blockDim.x) //simulates other blocks in case src.size > blockDim.x*2
            {
                //calculates which 2 indexes will be compared and swap
                int part = i / (len / 2);
                int s = part * len + (i & ((len / 2) - 1));
                int e = s + len / 2;

                if(e >= src.getSize())
                    break;

                //calculate the direction of swapping
                int monotonicSeqIdx = i / (monotonicSeqLen / 2);
                bool ascending = (monotonicSeqIdx & 1) != 0;
                if ((monotonicSeqIdx + 1) * monotonicSeqLen >= src.getSize()) //special case for parts with no "partner"
                    ascending = true;

                cmpSwap(src[s], src[e], ascending, Cmp);
            }
            __syncthreads();
        }
    }
}

/**
 * entrypoint for bitonicSort_Block
 * sorts @param arr in alternating order to create bitonic sequences
 * sharedMem has to be able to store at least blockDim.x*2 elements
 * */
template <typename Value, typename Function>
__global__ void bitoniSort1stStepSharedMemory(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr, const Function &Cmp)
{
    extern __shared__ int externMem[];
    int sharedMemLen = 2 * blockDim.x;
    int myBlockStart = blockIdx.x * sharedMemLen;
    int myBlockEnd = TNL::min(arr.getSize(), myBlockStart + sharedMemLen);

    if (blockIdx.x % 2 || blockIdx.x + 1 == gridDim.x)
        bitonicSort_Block(arr.getView(myBlockStart, myBlockEnd), arr.getView(myBlockStart, myBlockEnd), (Value *)externMem, Cmp);
    else
        bitonicSort_Block(arr.getView(myBlockStart, myBlockEnd), arr.getView(myBlockStart, myBlockEnd), (Value *)externMem,
                          [&] __cuda_callable__(const Value &a, const Value &b) { return Cmp(b, a); });
}

/**
 * entrypoint for bitonicSort_Block
 * sorts @param arr in alternating order to create bitonic sequences
 * doesn't use shared memory
 * */
template <typename Value, typename Function>
__global__ void bitoniSort1stStep(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr, const Function &Cmp)
{
    int myBlockStart = blockIdx.x * (2 * blockDim.x);
    int myBlockEnd = TNL::min(arr.getSize(), myBlockStart + (2 * blockDim.x));

    if (blockIdx.x % 2 || blockIdx.x + 1 == gridDim.x)
        bitonicSort_Block(arr.getView(myBlockStart, myBlockEnd), Cmp);
    else
        bitonicSort_Block(arr.getView(myBlockStart, myBlockEnd),
                          [&] __cuda_callable__(const Value &a, const Value &b) { return Cmp(b, a); });
}

//---------------------------------------------
template <typename Value, typename Function>
void bitonicSort(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> src, int begin, int end, const Function &Cmp)
{
    TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr = src.getView(begin, end);
    int paddedSize = closestPow2(arr.getSize());

    int threadsNeeded = arr.getSize() / 2 + (arr.getSize() % 2 != 0);

    const int maxThreadsPerBlock = 512;
    int threadPerBlock = maxThreadsPerBlock;
    int blocks = threadsNeeded / threadPerBlock + (threadsNeeded % threadPerBlock != 0);

    int sharedMemLen = threadPerBlock * 2;
    int sharedMemSize = sharedMemLen * sizeof(Value);

    //---------------------------------------------------------------------------------

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    //---------------------------------------------------------------------------------

    if (sharedMemSize <= deviceProp.sharedMemPerBlock)
        bitoniSort1stStepSharedMemory<<<blocks, threadPerBlock, sharedMemSize>>>(arr, Cmp);
    else
        bitoniSort1stStep<<<blocks, threadPerBlock>>>(arr, Cmp);

    for (int monotonicSeqLen = 2 * sharedMemLen; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2)
    {
        for (int len = monotonicSeqLen, partsInSeq = 1; len > 1; len /= 2, partsInSeq *= 2)
        {
            if (len > sharedMemLen)
            {
                bitonicMergeGlobal<<<blocks, threadPerBlock>>>(
                    arr, Cmp, monotonicSeqLen, len, partsInSeq);
            }
            else
            {
                if (sharedMemSize <= deviceProp.sharedMemPerBlock)
                {
                    bitonicMergeSharedMemory<<<blocks, threadPerBlock, sharedMemSize>>>(
                        arr, Cmp, monotonicSeqLen, len, partsInSeq);
                }
                else
                {
                    bitonicMerge<<<blocks, threadPerBlock>>>(
                        arr, Cmp, monotonicSeqLen, len, partsInSeq);
                }
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
    bitonicSort(arr, begin, end, [] __cuda_callable__(const Value &a, const Value &b) { return a < b; });
}

template <typename Value, typename Function>
void bitonicSort(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr, const Function &Cmp)
{
    bitonicSort(arr, 0, arr.getSize(), Cmp);
}

template <typename Value>
void bitonicSort(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr)
{
    bitonicSort(arr, [] __cuda_callable__(const Value &a, const Value &b) { return a < b; });
}

//---------------------------------------------
template <typename Value, typename Function>
void bitonicSort(std::vector<Value> &vec, int begin, int end, const Function &Cmp)
{
    TNL::Containers::Array<Value, TNL::Devices::Cuda> Arr(vec);
    auto view = Arr.getView();
    bitonicSort(view, begin, end, Cmp);

    TNL::Algorithms::MultiDeviceMemoryOperations<TNL::Devices::Host, TNL::Devices::Cuda>::
        copy(vec.data(), view.getData(), view.getSize());
}

template <typename Value>
void bitonicSort(std::vector<Value> &vec, int begin, int end)
{
    bitonicSort(vec, begin, end, [] __cuda_callable__(const Value &a, const Value &b) { return a < b; });
}

template <typename Value, typename Function>
void bitonicSort(std::vector<Value> &vec, const Function &Cmp)
{
    bitonicSort(vec, 0, vec.size(), Cmp);
}

template <typename Value>
void bitonicSort(std::vector<Value> &vec)
{
    bitonicSort(vec, [] __cuda_callable__(const Value &a, const Value &b) { return a < b; });
}

//---------------------------------------------
//---------------------------------------------

template <typename FETCH, typename CMP, typename SWAP>
__global__ void bitonicMergeGlobal(int size, FETCH Fetch,
                                   const CMP &Cmp, SWAP Swap,
                                   int monotonicSeqLen, int len, int partsInSeq)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int part = i / (len / 2); //computes which sorting block this thread belongs to

    //the index of 2 elements that should be compared and swapped
    int s = part * len + (i & ((len / 2) - 1));
    int e = s + len / 2;
    if (e >= size) //arr[e] is virtual padding and will not be exchanged with
        return;

    //calculate the direction of swapping
    int monotonicSeqIdx = part / partsInSeq;
    bool ascending = (monotonicSeqIdx & 1) != 0;
    if ((monotonicSeqIdx + 1) * monotonicSeqLen >= size) //special case for part with no "partner" to be merged with in next phase
        ascending = true;

    if (ascending == Cmp(Fetch(e), Fetch(s)))
        Swap(s, e);
}

template <typename FETCH, typename CMP, typename SWAP>
void bitonicSort(int begin, int end, FETCH Fetch, const CMP &Cmp, SWAP Swap)
{
    int size = end - begin;
    int paddedSize = closestPow2(size);

    int threadsNeeded = size / 2 + (size % 2 != 0);

    const int maxThreadsPerBlock = 512;
    int threadPerBlock = maxThreadsPerBlock;
    int blocks = threadsNeeded / threadPerBlock + (threadsNeeded % threadPerBlock != 0);

    auto fetchWithOffset =
        [=] __cuda_callable__(int i) {
            return Fetch(i + begin);
        };

    auto swapWithOffset =
        [=] __cuda_callable__(int i, int j) mutable {
            Swap(i + begin, j + begin);
        };

    for (int monotonicSeqLen = 2; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2)
    {
        for (int len = monotonicSeqLen, partsInSeq = 1; len > 1; len /= 2, partsInSeq *= 2)
        {
            bitonicMergeGlobal<<<blocks, threadPerBlock>>>(
                size, fetchWithOffset, Cmp, swapWithOffset, monotonicSeqLen, len, partsInSeq);
        }
    }
    cudaDeviceSynchronize();
}