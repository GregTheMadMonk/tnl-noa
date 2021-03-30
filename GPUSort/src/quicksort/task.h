#pragma once

struct TASK
{
    //start and end position of array to read and write from
    int partitionBegin, partitionEnd;
    //-----------------------------------------------
    //helper variables for blocks working on this task

    int depth;
    int pivotIdx;
    int dstBegin, dstEnd;
    int firstBlock, blockCount;//for workers read only values

    __cuda_callable__
    TASK(int begin, int end, int depth, int pivotIdx)
        : partitionBegin(begin), partitionEnd(end),
        depth(depth), pivotIdx(pivotIdx),
        dstBegin(-151561), dstEnd(-151561),
        firstBlock(-100), blockCount(-100)
        {}

    __cuda_callable__
    void initTask(int firstBlock, int blocks)
    {
        dstBegin= 0; dstEnd = partitionEnd - partitionBegin;
        this->firstBlock = firstBlock;
        blockCount = blocks;
    }

    TASK() = default;
};