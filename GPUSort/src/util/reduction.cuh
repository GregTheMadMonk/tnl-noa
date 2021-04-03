#pragma once
/**
 * https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
 * */

__device__ int warpReduceSum(int initVal)
{
    const unsigned int maskConstant = 0xffffffff; //not used
    for (unsigned int mask = warpSize / 2; mask > 0; mask >>= 1)
        initVal += __shfl_xor_sync(maskConstant, initVal, mask);

    return initVal;
}

__device__ int blockReduceSum(int val)
{
    static __shared__ int shared[32];
    int lane = threadIdx.x & (warpSize - 1);
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads(); 

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid == 0)
        val = warpReduceSum(val);

    if(threadIdx.x == 0)
        shared[0] = val;
    __syncthreads(); 

    return shared[0];
}

//-------------------------------------------------------------------------------

__device__ int warpInclusivePrefixSum(int value)
{
    int laneId = threadIdx.x & (32-1);

    #pragma unroll
    for (int i = 0; i < 5; i++) //iterates until x == 1<<4 == 16 which is half warpSize
    {
        int x = 1<<i;
        int n = __shfl_up_sync(0xffffffff, value, x);
        if ((laneId & (warpSize - 1)) >= x)
            value += n;
    }

    return value;
}

__device__ int blockInclusivePrefixSum(int value)
{
    static __shared__ int shared[32];
    int lane = threadIdx.x & (warpSize - 1);
    int wid = threadIdx.x / warpSize;

    int tmp = warpInclusivePrefixSum(value);

    if (lane == warpSize-1)
        shared[wid] = tmp;
    __syncthreads();

    int tmp2 = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0)
        shared[lane] = warpInclusivePrefixSum(tmp2) - tmp2;
    __syncthreads();
    
    tmp += shared[wid];
    return tmp;
}

//--------------------------------------------------------------------

template<typename Operator>
__device__ int warpCmpReduce(int initVal, const Operator & Cmp)
{
    const unsigned int maskConstant = 0xffffffff; //not used
    for (unsigned int mask = warpSize / 2; mask > 0; mask >>= 1)
        initVal = Cmp(initVal, __shfl_xor_sync(maskConstant, initVal, mask));

    return initVal;
}

template<typename Operator>
__device__ int blockCmpReduce(int val, const Operator & Cmp)
{
    static __shared__ int shared[32];
    int lane = threadIdx.x & (warpSize - 1);
    int wid = threadIdx.x / warpSize;

    val = warpCmpReduce(val, Cmp);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads(); 

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : shared[0];

    if (wid == 0)
        val = warpCmpReduce(val, Cmp);

    if(threadIdx.x == 0)
        shared[0] = val;
    __syncthreads(); 

    return shared[0];
}
