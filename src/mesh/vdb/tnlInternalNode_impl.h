#ifndef _TNLINTERNALNODE_IMPL_H_INCLUDED_
#define _TNLINTERNALNODE_IMPL_H_INCLUDED_

#include "tnlInternalNode.h"
#include "tnlLeafNode.h"
#include "tnlArea2D.h"
#include "tnlCircle2D.h"
#include "tnlBitmask.h"
#include "tnlVDBMath.h"
#include "tnlIterator2D.h"
#include "tnlBitmaskArray.h"

template< int LogX,
          int LogY >
tnlInternalNode< LogX, LogY >::tnlInternalNode( tnlArea2D* area,
                                                tnlCircle2D* circle,
                                                int X,
                                                int Y,
                                                int level )
{
    this->bitmaskArray = new tnlBitmaskArray< LogX * LogY >();
    this->area = area;
    this->circle = circle;
    this->X = X;
    this->Y = Y;
    this->level = level;
}

template< int LogX,
          int LogY >
void tnlInternalNode< LogX, LogY >::setNode( int splitX,
                                             int splitY,
                                             int depth )
{
    int depthX = splitX * tnlVDBMath::power( LogX, this->level );
    int depthY = splitY * tnlVDBMath::power( LogY, this->level );
    float stepX = ( float ) this->area->getLengthX() / depthX;
    float stepY = ( float ) this->area->getLengthY() / depthY;
    float startX = this->X * stepX;
    float endX = ( this->X + 1 ) * stepX;
    float startY = this->Y * stepY;
    float endY = ( this->Y + 1 ) * stepY;
    tnlIterator2D< LogX * LogY, LogX, LogY >* iter = 
                 new tnlIterator2D< LogX * LogY, LogX, LogY >( LogX,
                                                               LogY,
                                                               ( float ) ( endX - startX ) / LogX,
                                                               ( float ) ( endY - startY ) / LogY,
                                                               startX,
                                                               startY );
    iter->computeBitmaskArray( this->bitmaskArray, this->circle );
    this->setChildren( splitX, splitY, depth );
}

template< int LogX,
          int LogY >
void tnlInternalNode< LogX, LogY >::setChildren( int splitX,
                                                 int splitY,
                                                 int depth )
{
    for( int i = 0; i < LogX * LogY; i++ )
    {
        if( !this->bitmaskArray->getIthBitmask( i )->getState() )
            this->children[ i ] = NULL;
        else if( this->level < depth - 1 )
        {
            this->children[ i ] = new tnlInternalNode< LogX, LogY >( this->area,
                                                                     this->circle,
                                                                     this->bitmaskArray->getIthBitmask( i )->getX(),
                                                                     this->bitmaskArray->getIthBitmask( i )->getY(),
                                                                     this->level + 1 );
            this->children[ i ]->setNode( splitX, splitY, depth );
        }
        else
        {
            this->children[ i ] = new tnlLeafNode< LogX, LogY >( this->area,
                                                                 this->circle,
                                                                 this->bitmaskArray->getIthBitmask( i )->getX(),
                                                                 this->bitmaskArray->getIthBitmask( i )->getY(),
                                                                 this->level + 1 );
            this->children[ i ]->setNode( splitX, splitY, depth );
        }
    }
}

#endif // _TNLINTERNALNODE_IMPL_H_INCLUDED_
