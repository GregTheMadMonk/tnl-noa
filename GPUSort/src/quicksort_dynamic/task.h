#pragma once

struct TASK
{
    int begin, end;
    int stillWorkingCnt;
    
    __cuda_callable__
    TASK(int _begin, int _end, int blocks)
        : begin(_begin), end(_end), stillWorkingCnt(blocks){}

    __cuda_callable__
    TASK(){};
};