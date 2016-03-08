#ifndef _TNLBITMASKARRAY_IMPL_H_INCLUDED_
#define _TNLBITMASKARRAY_IMPL_H_INCLUDED_

#include <tnlBitmask.h>

template< unsigned size >
tnlBitmaskArray< size >::tnlBitmaskArray( unsigned size )
{
    this->size = size;
    this->bitmaskArray = new tnlBitmask*[size];
    // in this part all the partial masks should be computed
    // this could be handled by some iterator that knows
    // current depth of the n-tree


    	// TODO: vymyslet jak na to
}

template< unsigned size >
tnlBitmaskArray< size >::~tnlBitmaskArray()
{
    for( int i = 0; i < this->size; i++ )
        delete this->bitmaskArray[ i ];
    delete [] this->bitmaskArray;
}

#endif // _TNLBITMASKARRAY_IMPL_H_INCLUDED_
