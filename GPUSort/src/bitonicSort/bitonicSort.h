#pragma once
#include <TNL/Containers/Array.h>

//---------------------------------------------

// Inline PTX call to return index of highest non-zero bit in a word
static __device__ __forceinline__ unsigned int __btflo(unsigned int word)
{
    unsigned int ret;
    asm volatile("bfind.u32 %0, %1;"
                 : "=r"(ret)
                 : "r"(word));
    return ret;
}

__device__ int closestPow2_ptx(int bitonicLen)
{
    return 1 << (__btflo((unsigned)bitonicLen - 1U) + 1);
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

template <typename Value, typename CMP>
__cuda_callable__ void cmpSwap(Value &a, Value &b, bool ascending, const CMP &Cmp)
{
    if (ascending == Cmp(b, a))
        TNL::swap(a, b);
}

//---------------------------------------------

/**
 * this kernel simulates 1 exchange 
 * splits input arr that is bitonic into 2 bitonic sequences
 */
template <typename Value, typename CMP>
__global__ void bitonicMergeGlobal(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr,
                                   CMP Cmp,
                                   int monotonicSeqLen, int bitonicLen)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int part = i / (bitonicLen / 2); //computes which sorting block this thread belongs to

    //the index of 2 elements that should be compared and swapped
    int s = part * bitonicLen + (i & ((bitonicLen / 2) - 1));
    int e = s + bitonicLen / 2;
    if (e >= arr.getSize()) //arr[e] is virtual padding and will not be exchanged with
        return;

    int partsInSeq = monotonicSeqLen / bitonicLen;
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
template <typename Value, typename CMP>
__global__ void bitonicMergeSharedMemory(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr,
                                         CMP Cmp,
                                         int monotonicSeqLen, int bitonicLen)
{
    extern __shared__ int externMem[];
    Value *sharedMem = (Value *)externMem;

    int sharedMemLen = 2 * blockDim.x;

    //1st index and last index of subarray that this threadBlock should merge
    int myBlockStart = blockIdx.x * sharedMemLen;
    int myBlockEnd = TNL::min(arr.getSize(), myBlockStart + sharedMemLen);

    //copy from globalMem into sharedMem
    for (int i = threadIdx.x; myBlockStart + i < myBlockEnd; i += blockDim.x)
        sharedMem[i] = arr[myBlockStart + i];
    __syncthreads();

    //------------------------------------------
    //bitonic activity
    {
        //calculate the direction of swapping
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int part = i / (bitonicLen / 2);
        int partsInSeq = monotonicSeqLen / bitonicLen;
        int monotonicSeqIdx = part / partsInSeq;

        bool ascending = (monotonicSeqIdx & 1) != 0;
        //special case for parts with no "partner"
        if ((monotonicSeqIdx + 1) * monotonicSeqLen >= arr.getSize())
            ascending = true;
        //------------------------------------------

        //do bitonic merge
        for (; bitonicLen > 1; bitonicLen /= 2)
        {
            //calculates which 2 indexes will be compared and swap
            int part = threadIdx.x / (bitonicLen / 2);
            int s = part * bitonicLen + (threadIdx.x & ((bitonicLen / 2) - 1));
            int e = s + bitonicLen / 2;

            if (e < myBlockEnd - myBlockStart) //not touching virtual padding
                cmpSwap(sharedMem[s], sharedMem[e], ascending, Cmp);
            __syncthreads();
        }
    }

    //------------------------------------------

    //writeback to global memory
    for (int i = threadIdx.x; myBlockStart + i < myBlockEnd; i += blockDim.x)
        arr[myBlockStart + i] = sharedMem[i];
}

/**
 * simulates many layers of merge
 * turns input that is a bitonic sequence into 1 monotonic sequence
 * 
 * this user only operates on global memory, no shared memory is used
 * */
template <typename Value, typename CMP>
__global__ void bitonicMerge(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr,
                             CMP Cmp,
                             int monotonicSeqLen, int bitonicLen)
{
    //1st index and last index of subarray that this threadBlock should merge
    int myBlockStart = blockIdx.x * (2 * blockDim.x);
    int myBlockEnd = TNL::min(arr.getSize(), myBlockStart + (2 * blockDim.x));

    auto src = arr.getView(myBlockStart, myBlockEnd);

    //calculate the direction of swapping
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int part = i / (bitonicLen / 2);
    int partsInSeq = monotonicSeqLen / bitonicLen;
    int monotonicSeqIdx = part / partsInSeq;

    bool ascending = (monotonicSeqIdx & 1) != 0;
    //special case for parts with no "partner"
    if ((monotonicSeqIdx + 1) * monotonicSeqLen >= arr.getSize())
        ascending = true;
    //------------------------------------------

    //do bitonic merge
    for (; bitonicLen > 1; bitonicLen /= 2)
    {
        //calculates which 2 indexes will be compared and swap
        int part = threadIdx.x / (bitonicLen / 2);
        int s = part * bitonicLen + (threadIdx.x & ((bitonicLen / 2) - 1));
        int e = s + bitonicLen / 2;

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
template <typename Value, typename CMP>
__device__ void bitonicSort_Block(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> src,
                                  TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> dst,
                                  Value *sharedMem, const CMP &Cmp)
{
    //copy from globalMem into sharedMem
    for (int i = threadIdx.x; i < src.getSize(); i += blockDim.x)
        sharedMem[i] = src[i];
    __syncthreads();

    //------------------------------------------
    //bitonic activity
    {
        int paddedSize = closestPow2_ptx(src.getSize());

        for (int monotonicSeqLen = 2; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2)
        {
            for (int bitonicLen = monotonicSeqLen; bitonicLen > 1; bitonicLen /= 2)
            {
                for (int i = threadIdx.x;; i += blockDim.x) //simulates other blocks in case src.size > blockDim.x*2
                {
                    //calculates which 2 indexes will be compared and swap
                    int part = i / (bitonicLen / 2);
                    int s = part * bitonicLen + (i & ((bitonicLen / 2) - 1));
                    int e = s + bitonicLen / 2;

                    if (e >= src.getSize()) //touching virtual padding, the order dont swap
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
    for (int i = threadIdx.x; i < dst.getSize(); i += blockDim.x)
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
template <typename Value, typename CMP>
__device__ void bitonicSort_Block(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> src,
                                  const CMP &Cmp)
{
    int paddedSize = closestPow2_ptx(src.getSize());

    for (int monotonicSeqLen = 2; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2)
    {
        for (int bitonicLen = monotonicSeqLen; bitonicLen > 1; bitonicLen /= 2)
        {
            for (int i = threadIdx.x;; i += blockDim.x) //simulates other blocks in case src.size > blockDim.x*2
            {
                //calculates which 2 indexes will be compared and swap
                int part = i / (bitonicLen / 2);
                int s = part * bitonicLen + (i & ((bitonicLen / 2) - 1));
                int e = s + bitonicLen / 2;

                if (e >= src.getSize())
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
template <typename Value, typename CMP>
__global__ void bitoniSort1stStepSharedMemory(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr, CMP Cmp)
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
template <typename Value, typename CMP>
__global__ void bitoniSort1stStep(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr, CMP Cmp)
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
//---------------------------------------------
template <typename Value, typename CMP>
void bitonicSortWithShared(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> view, const CMP &Cmp,
                           int gridDim, int blockDim, int sharedMemLen, int sharedMemSize)
{
    int paddedSize = closestPow2(view.getSize());

    bitoniSort1stStepSharedMemory<<<gridDim, blockDim, sharedMemSize>>>(view, Cmp);
    //now alternating monotonic sequences with bitonicLenght of sharedMemLen

    // \/ has bitonicLength of 2 * sharedMemLen
    for (int monotonicSeqLen = 2 * sharedMemLen; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2)
    {
        for (int bitonicLen = monotonicSeqLen; bitonicLen > 1; bitonicLen /= 2)
        {
            if (bitonicLen > sharedMemLen)
            {
                bitonicMergeGlobal<<<gridDim, blockDim>>>(
                    view, Cmp, monotonicSeqLen, bitonicLen);
            }
            else
            {
                bitonicMergeSharedMemory<<<gridDim, blockDim, sharedMemSize>>>(
                    view, Cmp, monotonicSeqLen, bitonicLen);

                //simulates sorts until bitonicLen == 2 already, no need to continue this loop
                break;
            }
        }
    }
    cudaDeviceSynchronize();
}

//---------------------------------------------

template <typename Value, typename CMP>
void bitonicSort(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> view,
                 const CMP &Cmp,
                 int gridDim, int blockDim)

{
    int paddedSize = closestPow2(view.getSize());

    for (int monotonicSeqLen = 2; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2)
    {
        for (int bitonicLen = monotonicSeqLen; bitonicLen > 1; bitonicLen /= 2)
        {
            bitonicMergeGlobal<<<gridDim, blockDim>>>(view, Cmp, monotonicSeqLen, bitonicLen);
        }
    }
    cudaDeviceSynchronize();
}

//---------------------------------------------
template <typename Value, typename CMP>
void bitonicSort(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> src, int begin, int end, const CMP &Cmp)
{
    auto view = src.getView(begin, end);

    int threadsNeeded = view.getSize() / 2 + (view.getSize() % 2 != 0);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const int maxThreadsPerBlock = 512;

    int sharedMemLen = maxThreadsPerBlock * 2;
    int sharedMemSize = sharedMemLen * sizeof(Value);

    if (sharedMemSize <= deviceProp.sharedMemPerBlock)
    {
        int blockDim = maxThreadsPerBlock;
        int gridDim = threadsNeeded / blockDim + (threadsNeeded % blockDim != 0);
        bitonicSortWithShared(view, Cmp, gridDim, blockDim, sharedMemLen, sharedMemSize);
    }
    else if (sharedMemSize / 2 <= deviceProp.sharedMemPerBlock)
    {
        int blockDim = maxThreadsPerBlock / 2; //256
        int gridDim = threadsNeeded / blockDim + (threadsNeeded % blockDim != 0);
        sharedMemSize /= 2;
        sharedMemLen /= 2;
        bitonicSortWithShared(view, Cmp, gridDim, blockDim, sharedMemLen, sharedMemSize);
    }
    else
    {
        int gridDim = threadsNeeded / maxThreadsPerBlock + (threadsNeeded % maxThreadsPerBlock != 0);
        bitonicSort(view, Cmp, gridDim, maxThreadsPerBlock);
    }
}

//---------------------------------------------

template <typename Value, typename CMP>
void bitonicSort(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr, int begin, int end)
{
    bitonicSort(arr, begin, end, [] __cuda_callable__(const Value &a, const Value &b) { return a < b; });
}

template <typename Value, typename CMP>
void bitonicSort(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr, const CMP &Cmp)
{
    bitonicSort(arr, 0, arr.getSize(), Cmp);
}

template <typename Value>
void bitonicSort(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr)
{
    bitonicSort(arr, [] __cuda_callable__(const Value &a, const Value &b) { return a < b; });
}

//---------------------------------------------
template <typename Value, typename CMP>
void bitonicSort(std::vector<Value> &vec, int begin, int end, const CMP &Cmp)
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

template <typename Value, typename CMP>
void bitonicSort(std::vector<Value> &vec, const CMP &Cmp)
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
__global__ void bitonicMergeGlobal(int size, FETCH Fetch, CMP Cmp, SWAP Swap,
                                   int monotonicSeqLen, int bitonicLen)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int part = i / (bitonicLen / 2); //computes which sorting block this thread belongs to

    //the index of 2 elements that should be compared and swapped
    int s = part * bitonicLen + (i & ((bitonicLen / 2) - 1));
    int e = s + bitonicLen / 2;
    if (e >= size) //arr[e] is virtual padding and will not be exchanged with
        return;

    //calculate the direction of swapping
    int partsInSeq = monotonicSeqLen / bitonicLen;
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
    int threadsPerBlock = maxThreadsPerBlock;
    int blocks = threadsNeeded / threadsPerBlock + (threadsNeeded % threadsPerBlock != 0);

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
        for (int bitonicLen = monotonicSeqLen; bitonicLen > 1; bitonicLen /= 2)
        {
            bitonicMergeGlobal<<<blocks, threadsPerBlock>>>(
                size, fetchWithOffset, Cmp, swapWithOffset,
                monotonicSeqLen, bitonicLen);
        }
    }
    cudaDeviceSynchronize();
}