#ifndef _TNLNODE_IMPL_H_INCLUDED_
#define _TNLNODE_IMPL_H_INCLUDED_

#include "tnlNode.h"
#include "tnlIterator2D.h"
#include "tnlVDBMath.h"

template< int LogX,
          int LogY >
tnlNode< LogX, LogY >::tnlNode( tnlArea2D* area,
                                tnlCircle2D* circle,
                                int X,
                                int Y,
                                int level )
{
    this->area = area;
    this->circle = circle;
    this->level = level;
    this->X = X;
    this->Y = Y;
}

template< int LogX,
          int LogY >
int tnlNode< LogX, LogY >::getLevel()
{
    return this->level;
}

template< int LogX,
          int LogY >
void tnlNode< LogX, LogY >::setNode( int splitX,
                                     int splitY,
                                     tnlBitmaskArray< LogX * LogY >* bitmaskArray )
{

    int depthX = splitX * tnlVDBMath::power( LogX, this->level - 1 );
    int depthY = splitY * tnlVDBMath::power( LogY, this->level - 1 );
    float stepX = ( float ) this->area->getLengthX() / depthX;
    float stepY = ( float ) this->area->getLengthY() / depthY;
    float startX = this->X * stepX;
    float endX = ( this->X + 1 ) * stepX;
    float startY = this->Y * stepY;
    float endY = ( this->Y + 1 ) * stepY;
    float dx = ( endX - startX ) / LogX;
    float dy = ( endY - startY ) / LogY;
    for( int i = 0; i < LogY; i++ )
        for( int j = 0; j < LogX; j++ )
        {
            float x1 = startX + j * dx;
            float x2 = startX + ( j + 1 ) * dx;
            float y1 = startY + i * dy;
            float y2 = startY + ( i + 1 ) * dy;
            bool state = this->circle->isIntercept( x1, x2, y1, y2 );
            int posX = this->X * LogX + j;
            int posY = this->Y * LogY + i;
            tnlBitmask* bitmask = new tnlBitmask( state, posX, posY );
            bitmaskArray->setIthBitmask( i * LogX + j, bitmask );
        }
}

template< int LogX,
          int LogY >
tnlNode< LogX, LogY >::~tnlNode()
{
    this->area = NULL;
    this->circle = NULL;
}

#endif // _TNLNODE_IMPL_H_INCLUDED_
