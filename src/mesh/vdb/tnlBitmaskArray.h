#ifndef _TNLBITMASKARRAY_H_INCLUDED_
#define _TNLBITMASKARRAY_H_INCLUDED_

#include "tnlBitmask.h"

template< unsigned size >
class tnlBitmaskArray
{
public:
    tnlBitmaskArray();

    unsigned getSize();

    void setIthBitmask( unsigned i,
                        tnlBitmask bitmask );

    tnlBitmask* getIthBitmask( unsigned i );

    ~tnlBitmaskArray();

private:
    tnlBitmask* bitmaskArray[ size ];
    unsigned length;
};

#include "tnlBitmaskArray_impl.h"
#endif // _TNLBITMASKARRAY_H_INCLUDED_
