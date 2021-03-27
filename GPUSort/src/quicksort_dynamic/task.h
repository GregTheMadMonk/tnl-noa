#pragma once

struct TASK
{
    int begin, end;
    
    __cuda_callable__
    TASK(int _begin, int _end)
        : begin(_begin), end(_end){}

    __cuda_callable__
    TASK(){};
};