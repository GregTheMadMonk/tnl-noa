#ifndef _TNLITERATOR2D_IMPL_H_INCLUDED_
#define _TNLITERATOR2D_IMPL_H_INCLUDED_

#include <tnlIterator2D.h>

tnlIterator2D::tnlIterator2D( unsigned logX,
                              unsigned logY,
                              unsigned parentX = 0,
                              unsigned parentY = 0,
                              unsigned level = 0 );
{
    this->level = level;
    this->parentX = parentX;
    this->parentY = parentY;
    this->logX = logX;
    this->logY = logY;
}

void tnlIterator2D::computeBitmaskArray( tnlBitmaskArray bitmaskArray,
                                         unsigned size );
{
    // TODO
}


#endif // _TNLITERATOR2D_IMPL_H_INCLUDED_
