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