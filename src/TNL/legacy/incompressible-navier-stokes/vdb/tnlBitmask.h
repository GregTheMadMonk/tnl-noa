#ifndef _TNLBITMASK_H_INCLUDED_
#define _TNLBITMASK_H_INCLUDED_

#include <cstdint>

class tnlBitmask
{
public:
    tnlBitmask( bool state, unsigned x, unsigned y );

    tnlBitmask( tnlBitmask* bitmask );
    
    bool getState();
    
    unsigned getX();
    
    unsigned getY();
    
    uint64_t getBitmask();

    ~tnlBitmask(){};
    
private:
    uint64_t bitmask;
};

#include "tnlBitmask_impl.h"
#endif //_TNLBITMASK_H_INCLUDED_
