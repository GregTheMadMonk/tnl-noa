#pragma once

struct TASK
{
    int arrBegin, arrEnd;//start and end position of array to read from
    int auxBeginIdx, auxEndIdx; //start and end position of still available memory to write into
    int firstBlock, blockCount;
    int stillWorkingCnt;//shared counter of blocks working together(how many are still working)
    

    __cuda_callable__
    TASK(int srcBegin, int srcEnd, int destBegin, int destEnd)
        : arrBegin(srcBegin), arrEnd(srcEnd),
        auxBeginIdx(destBegin), auxEndIdx(destEnd),
        firstBlock(-1), blockCount(-1), stillWorkingCnt(-1)
        {}

    __cuda_callable__ void setBlocks(int blocks)
    {
        blockCount = stillWorkingCnt = blocks;
    }
    TASK() = default;

};