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

__device__ int warpInclusivePrefixSum(int value)
{
    int laneId = threadIdx.x & 0x1f;

    #pragma unroll
    for (int i = 1; i*2 <= 32; i *= 2)//32 here is warp size
    {
        int n = __shfl_up_sync(0xffffffff, value, i);
        if ((laneId & (warpSize - 1)) >= i)
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