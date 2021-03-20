#pragma once

#include <TNL/Containers/Array.h>

template <typename Value, typename Device>
__device__ void countElem(TNL::Containers::ArrayView<Value, Device> src,
                          int myBegin, int myEnd,
                          int &smaller, int &bigger,
                          const Value &pivot)
{
    for (int i = myBegin + threadIdx.x; i < myEnd; i += blockDim.x)
    {
        int data = src[i];
        if (data < pivot)
            smaller++;
        else if (data > pivot)
            bigger++;
    }
}

template <typename Value, typename Device>
__device__ void copyData(TNL::Containers::ArrayView<Value, Device> src,
                         int myBegin, int myEnd,
                         TNL::Containers::ArrayView<Value, Device> dst,
                         int smallerStart, int biggerStart,
                         const Value &pivot)

{
    for (int i = myBegin + threadIdx.x; i < myEnd; i += blockDim.x)
    {
        int data = src[i];
        if (data < pivot)
            dst[smallerStart++] = data;
        else if (data > pivot)
            dst[biggerStart++] = data;
    }
}

__device__ void calcBlocksNeeded(int elemLeft, int elemRight, int &blocksLeft, int &blocksRight)
{
    int minElemPerBlock = blockDim.x*2;
    blocksLeft = elemLeft / minElemPerBlock + (elemLeft% minElemPerBlock != 0);
    blocksRight = elemRight / minElemPerBlock + (elemRight% minElemPerBlock != 0);

    
    int totalSets = blocksLeft + blocksRight;
    if(totalSets<= gridDim.x)
        return;

    int multiplier = 1.*gridDim.x / totalSets + 1;
    minElemPerBlock *= multiplier;

    blocksLeft = elemLeft / minElemPerBlock + (elemLeft% minElemPerBlock != 0);
    blocksRight = elemRight / minElemPerBlock + (elemRight% minElemPerBlock != 0);
    
}

template <typename Value, typename Device, typename Function>
__device__ Value pickPivot(TNL::Containers::ArrayView<Value, Device> src, const Function & Cmp)
{
    return src[0];
    //return src[src.getSize()-1];

    /*
    if(src.getSize() ==1)
        return src[0];
    
    Value a = src[0], b = src[src.getSize()/2], c = src[src.getSize() - 1];

    if(Cmp(a, b)) // ..a..b..
    {
        if(Cmp(b, c))// ..a..b..c
            return b;
        else if(Cmp(c, a))//..c..a..b..
            return a;
        else //..a..c..b..
            return c;
    }
    else //..b..a..
    {
        if(Cmp(a, c))//..b..a..c
            return a;
        else if(Cmp(c, b))//..c..b..a..
            return b;
        else //..b..c..a..
            return c;
    }
    */
}