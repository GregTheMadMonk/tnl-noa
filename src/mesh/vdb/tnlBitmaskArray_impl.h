#ifndef _TNLBITMASKARRAY_IMPL_H_INCLUDED_
#define _TNLBITMASKARRAY_IMPL_H_INCLUDED_

#include <cassert>
#include "tnlBitmask.h"
#include "tnlBitmaskArray.h"

template< unsigned size >
tnlBitmaskArray< size >::tnlBitmaskArray()
{
    this->length = size;
}

template< unsigned size >
unsigned tnlBitmaskArray< size >::getSize()
{
    return this->length;
}

template< unsigned size >
void tnlBitmaskArray< size >::setIthBitmask( unsigned i,
                                             tnlBitmask bitmask )
{
    assert( i < size );
    this->bitmaskArray[ i ] = new tnlBitmask( bitmask );
}

template< unsigned size >
tnlBitmask* tnlBitmaskArray< size >::getIthBitmask( unsigned i )
{
    assert( i < size );
    return this->bitmaskArray[ i ];
}

template< unsigned size >
tnlBitmaskArray< size >::~tnlBitmaskArray()
{
    for( int i = 0; i < this->length; i++ )
        delete this->bitmaskArray[ i ];
    delete this->bitmaskArray;
}

#endif // _TNLBITMASKARRAY_IMPL_H_INCLUDED_
