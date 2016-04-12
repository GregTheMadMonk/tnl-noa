#ifndef _TNLITERATOR2D_IMPL_H_INCLUDED_
#define _TNLITERATOR2D_IMPL_H_INCLUDED_

#include "tnlIterator2D.h"
#include "tnlBitmaskArray.h"
#include "tnlBitmask.h"
#include "tnlCircle2D.h"
#include <fstream>

template< unsigned size,
          int LogX,
          int LogY >
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
          int LogY >
void tnlIterator2D< size, LogX, LogY >::computeBitmaskArray( tnlBitmaskArray< size >* bitmaskArray,
                                                             tnlCircle2D* circle,
                                                             int posX,
                                                             int posY )
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
            int X = posX * cellsX + j;
            int Y = posY * cellsY + i;
            tnlBitmask* bitmask = new tnlBitmask( state, X, Y );
            bitmaskArray->setIthBitmask( i * this->cellsX + j, bitmask );
        }
}

template< unsigned size,
          int LogX,
          int LogY >
void tnlIterator2D< size, LogX, LogY >::dumpIntoFile( tnlBitmaskArray< size >* bitmaskArray,
                                                      fstream& file,
                                                      int level )
{
    for( int i = 0; i < this->cellsY; i++ )
        for( int j = 0; j < this->cellsX; j++ )
        {
            //float x1 = this->startX + j * this->stepX;
            //float x2 = this->startX + ( j + 1 ) * this->stepX;
            //float y1 = this->startY + i * this->stepY;
            //float y2 = this->startY + ( i + 1 ) * this->stepY;
            int x = bitmaskArray->getIthBitmask( i * this->cellsX + j )->getX();
            int y = bitmaskArray->getIthBitmask( i * this->cellsX + j )->getY();
            bool state = bitmaskArray->getIthBitmask( i * this->cellsX + j )->getState();
            file << "x = " << x <<
                    ", y = " << y <<
                    ", state = " << state << 
                    ", level = " << level << std::endl;
        }
}

#endif // _TNLITERATOR2D_IMPL_H_INCLUDED_
