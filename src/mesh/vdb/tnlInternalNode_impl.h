#ifndef _TNLINTERNALNODE_IMPL_H_INCLUDED_
#define _TNLINTERNALNODE_IMPL_H_INCLUDED_

#include "tnlInternalNode.h"
#include "tnlArea2D.h"
#include "tnlCircle2D.h"
#include "tnlBitmask.h"
#include "tnlVDBMath.h"

template< int LogX,
          int LogY = LogX >
tnlInternalNode< LogX, LogY >::tnlInternalNode( tnlArea2D* area,
                                                tnlCircle2D* circle,
                                                tnlBitmask* coordinates,
                                                int level )
{
    this->area = area;
    this->circle = circle;
    this->coordinates = coordinates;
    this->level = level;
}

template< int LogX,
          int LogY = LogX >
void tnlInternalNode< LogX, LogY >::setNode( int splitX,
                                             int splitY,
                                             int depth )
{
    int depthX = splitX * tnlVDBMath::power( LogX, level );
    int depthY = splitY * tnlVDBMath::power( LogY, level );
    float stepX = ( float ) this->area->getLengthX() / depthX;
    float stepY = ( float ) this->area->getLengthY() / depthY;
    float startX = this->coordinates->getX() * stepX;
    float endX = ( this->coordinates->getX() + 1 ) * stepX;
    float startY = this->coordinates->getY() * stepY;
    float endY = ( this->coordinates->getY() + 1 ) * stepY;
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
          int LogY = LogX >
void tnlInternalNode< LogX, LogY >::setChildren( int splitX,
                                                 int splitY,
                                                 int depth )
{
    for( int i = 0; i < LogX * LogY; i++ )
    {
        if( !this->bitmaskArray->getIthMask()->getState() )
            this->children[ i ] = NULL;
        else if( this->level < depth - 1 )
        {
            this->children[ i ] = new tnlInternalNode( this->area,
                                                       this->circle,
                                                       this->bitmaskArray->getIthBitmask( i ),
                                                       this->level + 1 );
            this->children[ i ]->setNode( splitX, splitY, depth );
        }
        else
        {
            this->children[ i ] = new tnlLeafNode( this->area,
                                                   this->circle,
                                                   this->bitmaskArray->getIthBitmask( i ),
                                                   this->level + 1 );
            this->children[ i ]->setNode( splitX, splitY, depth );
        }
    }
}

#endif // _TNLINTERNALNODE_IMPL_H_INCLUDED_
