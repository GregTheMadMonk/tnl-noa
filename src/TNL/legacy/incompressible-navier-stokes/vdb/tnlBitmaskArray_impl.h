#ifndef _TNLBITMASKARRAY_IMPL_H_INCLUDED_
#define _TNLBITMASKARRAY_IMPL_H_INCLUDED_

#include <cassert>
#include "tnlBitmask.h"
#include "tnlBitmaskArray.h"

template< unsigned Size >
tnlBitmaskArray< Size >::tnlBitmaskArray()
{
    this->length = Size;
}

template< unsigned Size >
unsigned tnlBitmaskArray< Size >::getSize()
{
    return this->length;
}

template< unsigned Size >
void tnlBitmaskArray< Size >::setIthBitmask( unsigned i,
                                             tnlBitmask bitmask )
{
    assert( i < Size );
    this->bitmaskArray[ i ] = new tnlBitmask( bitmask );
}

template< unsigned Size >
tnlBitmask* tnlBitmaskArray< Size >::getIthBitmask( unsigned i )
{
    assert( i < Size );
    return this->bitmaskArray[ i ];
}

template< unsigned Size >
tnlBitmaskArray< Size >::~tnlBitmaskArray()
{
    for( int i = 0; i < this->length; i++ )
        delete this->bitmaskArray[ i ];
    delete this->bitmaskArray;
}

#endif // _TNLBITMASKARRAY_IMPL_H_INCLUDED_
