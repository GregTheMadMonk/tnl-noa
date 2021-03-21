#pragma once

struct TASK
{
    //start and end position of array to read and write from
    int partitionBegin, partitionEnd;
    //-----------------------------------------------
    //helper variables for blocks working on this task

    int depth;
    int dstBegin, dstEnd;
    int firstBlock, blockCount;//for workers read only values
    int stillWorkingCnt;//shared counter of blocks working together(how many are still working)

    __cuda_callable__
    TASK(int begin, int end, int depth)
        : partitionBegin(begin), partitionEnd(end),
        depth(depth),
        dstBegin(-151561), dstEnd(-151561),
        firstBlock(-100), blockCount(-100), stillWorkingCnt(-100)
        {}

    __cuda_callable__
    void initTask(int firstBlock, int blocks)
    {
        dstBegin= 0; dstEnd = partitionEnd - partitionBegin;
        this->firstBlock = firstBlock;
        blockCount = stillWorkingCnt = blocks;
    }

    TASK() = default;
};