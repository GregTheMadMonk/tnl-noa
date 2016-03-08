#ifndef _TNLBITMASKARRAY_H_INCLUDED_
#define _TNLBITMASKARRAY_H_INCLUDED_

#include <tnlBitmask.h>

template< unsigned size >
class tnlBitmaskArray
{
public:
    tnlBitmaskArray( unsigned size );

    

    ~tnlBitmaskArray();

private:
    tnlBitmask** bitmaskArray;
    unsigned size;
};

#include <tnlBitmaskArray_impl.h>
#endif // _TNLBITMASKARRAY_H_INCLUDED_
