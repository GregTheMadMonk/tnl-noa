#ifndef _TNLLEAFNODE_IMPL_H_INCLUDED_
#define _TNLLEAFNODE_IMPL_H_INCLUDED_

#include "tnlLeafNode.h"
#include "tnlArea2D.h"
#include "tnlCircle2D.h"
#include "tnlBitmask.h"
#include "tnlIterator2D.h"

template< int LogX,
          int LogY = LogX >
tnlLeafNode< LogX, LogY >::tnlLeafNode( tnlArea2D* area,
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
tnlLeafNode< LogX, LogY >::setNode( int splitX,
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
}

template< int LogX,
          int LogY = LogX >
tnlLeafNode< LogX, LogY >::~tnlLeafNode()
{
    this->area = NULL;
    this->circle = NULL;
    this->coordinates = NULL;
}

#endif // _TNLLEAFNODE_IMPL_H_INCLUDED_
