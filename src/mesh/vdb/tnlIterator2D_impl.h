#ifndef _TNLITERATOR2D_IMPL_H_INCLUDED_
#define _TNLITERATOR2D_IMPL_H_INCLUDED_

#include "tnlIterator2D.h"
#include "tnlBitmaskArray.h"
#include "tnlBitmask.h"
#include "tnlCircle2D.h"

template< unsigned size,
          int LogX,
          int LogY = LogX >
tnlIterator2D< size, LogX, LogY >::tnlIterator2D( unsigned cellsX,
                                      unsigned cellsY,
                                      float stepX,
                                      float stepY,
                                      float startX,
                                      float startY )
{
    this->cellsX = cellsX;
    this->cellsY = cellsY;
    this->stepX = stepX;
    this->stepY = stepY;
    this->startX = startX;
    this->startY = startY;
}

template< unsigned size,
          int LogX,
          int LogY = LogX >
void tnlIterator2D< size, LogX, LogY >::computeBitmaskArray( tnlBitmaskArray< size >* bitmaskArray,
                                                 tnlCircle2D* circle )
{
    // yeah, in matrix, i like to iterate over rows first
    for( int i = 0; i < this->cellsY; i++ )
        for( int j = 0; j < this->cellsX; j++ )
        {
            float x1 = this->startX + j * this->stepX;
            float x2 = this->startX + ( j + 1 ) * this->stepX;
            float y1 = this->startY + i * this->stepY;
            float y2 = this->startY + ( i + 1 ) * this->stepY;
            bool state = circle->isIntercept( x1, x2, y1, y2 );

            bitmaskArray->setIthBitmask( i * this->cellsX + j, new tnlBitmask( state, j, i ) );
        }
}

#endif // _TNLITERATOR2D_IMPL_H_INCLUDED_
