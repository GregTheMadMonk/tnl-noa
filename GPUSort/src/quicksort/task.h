#pragma once

struct TASK
{
    //start and end position of array to read and write from
    int partitionBegin, partitionEnd;

    //-----------------------------------------------
    //helper variables for blocks working on this task

    int dstBegin, dstEnd;
    int firstBlock, blockCount;//for workers read only values
    int stillWorkingCnt;//shared counter of blocks working together(how many are still working)

    __cuda_callable__
    TASK(int begin, int end)
        : partitionBegin(begin), partitionEnd(end),
        dstBegin(0), dstEnd(end-begin),
        firstBlock(-100), blockCount(-100), stillWorkingCnt(-100)
        {}

    __cuda_callable__
    void setBlocks(int blocks)
    {
        blockCount = stillWorkingCnt = blocks;
    }

    TASK() = default;
};