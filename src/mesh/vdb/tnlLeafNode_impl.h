#ifndef _TNLLEAFNODE_IMPL_H_INCLUDED_
#define _TNLLEAFNODE_IMPL_H_INCLUDED_

#include "tnlLeafNode.h"
#include "tnlArea2D.h"
#include "tnlCircle2D.h"
#include "tnlBitmask.h"
#include "tnlIterator2D.h"
#include "tnlVDBMath.h"
#include <iostream>
#include "tnlBitmaskArray.h"

template< int LogX,
          int LogY >
tnlLeafNode< LogX, LogY >::tnlLeafNode( tnlArea2D* area,
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
void tnlLeafNode< LogX, LogY >::setNode( int splitX,
                                         int splitY,
                                         int depth )
{
    std::cout << "tnlLeafNode::setNode" << std::endl;
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
}

template< int LogX,
          int LogY >
tnlLeafNode< LogX, LogY >::~tnlLeafNode()
{
    this->area = NULL;
    this->circle = NULL;
}

#endif // _TNLLEAFNODE_IMPL_H_INCLUDED_
