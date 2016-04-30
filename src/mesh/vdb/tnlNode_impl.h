#ifndef _TNLNODE_IMPL_H_INCLUDED_
#define _TNLNODE_IMPL_H_INCLUDED_

#include "tnlNode.h"
#include "tnlVDBMath.h"

template< typename Real,
          typename Index,
          Index LogX,
          Index LogY >
tnlNode< Real, Index, LogX, LogY >::tnlNode( tnlArea2D< Real >* area,
                                             tnlCircle2D< Real >* circle,
                                             Index X,
                                             Index Y,
                                             Index level )
{
    this->area = area;
    this->circle = circle;
    this->level = level;
    this->X = X;
    this->Y = Y;
}

template< typename Real,
          typename Index,
          Index LogX,
          Index LogY >
Index tnlNode< Real, Index, LogX, LogY >::getLevel()
{
    return this->level;
}

template< typename Real,
          typename Index,
          Index LogX,
          Index LogY >
void tnlNode< Real, Index, LogX, LogY >::setNode( Index splitX,
                                                  Index splitY,
                                                  tnlBitmaskArray< LogX * LogY >* bitmaskArray )
{

    Index depthX = splitX * tnlVDBMath< Index >::power( LogX, this->level - 1 );
    Index depthY = splitY * tnlVDBMath< Index >::power( LogY, this->level - 1 );
    Real stepX = ( Real ) this->area->getLengthX() / depthX;
    Real stepY = ( Real ) this->area->getLengthY() / depthY;
    Real startX = this->X * stepX;
    Real endX = ( this->X + 1 ) * stepX;
    Real startY = this->Y * stepY;
    Real endY = ( this->Y + 1 ) * stepY;
    Real dx = ( endX - startX ) / LogX;
    Real dy = ( endY - startY ) / LogY;
    for( Index i = 0; i < LogY; i++ )
        for( Index j = 0; j < LogX; j++ )
        {
            Real x1 = startX + j * dx;
            Real x2 = startX + ( j + 1 ) * dx;
            Real y1 = startY + i * dy;
            Real y2 = startY + ( i + 1 ) * dy;
            bool state = this->circle->isIntercept( x1, x2, y1, y2 );
            Index posX = this->X * LogX + j;
            Index posY = this->Y * LogY + i;
            tnlBitmask* bitmask = new tnlBitmask( state, posX, posY );
            bitmaskArray->setIthBitmask( i * LogX + j, bitmask );
        }
}

template< typename Real,
          typename Index,
          Index LogX,
          Index LogY >
tnlNode< Real, Index, LogX, LogY >::~tnlNode()
{
    this->area = NULL;
    this->circle = NULL;
}

#endif // _TNLNODE_IMPL_H_INCLUDED_
