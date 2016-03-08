#ifndef _TNLBITMASK_IMPL_H_INCLUDED_
#define _TNLBITMASK_IMPL_H_INCLUDED_

#include <iostream>
#include <cstdint>
#include "tnlBitmask.h"

using namespace std;

tnlBitmask::tnlBitmask( bool state,
                        unsigned x,
                        unsigned y )
/*
  variables x and y have at most 30 active bits
*/                        
{
    uint64_t state64 = state;
    uint64_t x64 = x;
    x64 <<= 4;
    uint64_t y64 = y;
    y64 <<= 34;
    this->bitmask = x64 | y64 | state64;
}                        

bool tnlBitmask::getState()
{
    return this->bitmask & 1;
}

unsigned tnlBitmask::getX()
{
    unsigned mask = 3 << 30;
    unsigned x = this->bitmask >> 4;
    return ( unsigned ) ( x & ( ~mask ) );
}

unsigned tnlBitmask::getY()
{
    unsigned mask = 3 << 30;
    uint64_t y = this->bitmask >> 34;
    return ( unsigned ) ( y & ( ~mask ) );
}

#endif //_TNLBITMASK_IMPL_H_INCLUDED_
