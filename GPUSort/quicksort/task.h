#pragma once

struct TASK
{
    int arrBegin, arrEnd;//start and end position of array to read from
    int auxBeginIdx, auxEndIdx; //start and end position of still available memory to write into
    int pivot;
    int firstBlock, blockCount; //shared counter of blocks working together(how many are still working)
    

    __cuda_callable__
    TASK(int srcBegin, int srcEnd, int destBegin, int destEnd, int pivot)
        : arrBegin(srcBegin), arrEnd(srcEnd),
        auxBeginIdx(destBegin), auxEndIdx(destEnd),
        pivot(pivot),
        firstBlock(-1), blockCount(-1)
        {}
    TASK() = default;

};