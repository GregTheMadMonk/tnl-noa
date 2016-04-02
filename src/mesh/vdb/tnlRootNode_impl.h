#ifndef _TNLROOTNODE_IMPL_H_INCLUDED_
#define _TNLROOTNODE_IMPL_H_INCLUDED_

#include <iostream>
#include "tnlArea2D.h"
#include "tnlIterator2D.h"
#include "tnlRootNode.h"
#include "tnlCircle2D.h"

template< unsigned size,
          int LogX,
          int LogY = LogX >
tnlRootNode< size, LogX, LogY >::tnlRootNode( tnlArea2D* area,
                                  tnlCircle2D* circle,
                                  unsigned nodesX,
                                  unsigned nodesY,
                                  unsigned treeDepth )
{
    this->area = area;
    this->circle = circle;
    this->nodesX = nodesX;
    this->nodesY = nodesY;
    this->bitmaskArray = new tnlBitmaskArray< size, LogX, LogY >();
    this->level = 0;
}

template< unsigned size, 
          int LogX,
          int LogY = LogX >
void tnlRootNode< size, LogX, LogY >::setNode()
{
    float stepX = ( this->area->getEndX() - this->area->getStartX() ) / this->nodesX;
    float stepY = ( this->area->getEndY() - this->area->getStartY() ) / this->nodesY;
    tnlIterator2D< size, LogX, LogY >* iter = new tnlIterator2D< size, LogX, LogY >( this->nodesX,
                                                             this->nodesY,
                                                             stepX,
                                                             stepY,
                                                             this->area->getStartX(),
                                                             this->area->getStartY() );
    iter->computeBitmaskArray( this->bitmaskArray, this->circle );
}

template< unsigned size,
          int LogX,
          int LogY = LogX >
void tnlRootNode< size, LogX, LogY >::createTree()
{
    this->setNode(); // first we need to create root node
}

template< unsigned size,
          int LogX,
          int LogY = LogX >
void tnlRootNode< size, LogX, LogY >::printStates()
{
    for( int i = 0; i < size; i++ )
        std::cout << this->bitmaskArray->getIthBitmask( i )->getState() << std::endl;
}

template< unsigned size,
          int LogX,
          int LogY = LogX >
tnlRootNode< size, LogX, LogY >::~tnlRootNode()
{
    delete this->area;
    delete this->circle;
    delete this->bitmaskArray;
}

#endif // _TNLROOTNODE_IMPL_H_INCLUDED_
