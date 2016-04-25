#ifndef _TNLLEAFNODE_IMPL_H_INCLUDED_
#define _TNLLEAFNODE_IMPL_H_INCLUDED_

#include "tnlLeafNode.h"
#include <iostream>
#include <iomanip>

template< int LogX,
          int LogY >
tnlLeafNode< LogX, LogY >::tnlLeafNode( tnlArea2D* area,
                                        tnlCircle2D* circle,
                                        int X,
                                        int Y,
                                        int level )
: tnlNode< LogX, LogY >::tnlNode( area, circle, X, Y, level )
{
    this->bitmaskArray = new tnlBitmaskArray< LogX * LogY >();
}

template< int LogX,
          int LogY >
void tnlLeafNode< LogX, LogY >::setNode( int splitX,
                                         int splitY,
                                         int depth )
{
    tnlNode< LogX, LogY >::setNode( splitX, splitY, this->bitmaskArray );
}

template< int LogX,
          int LogY >
void tnlLeafNode< LogX, LogY >::write( fstream& file,
                                       int level )
{
    for( int i = 0; i < LogX * LogY; i++ )
    {
        int x = this->bitmaskArray->getIthBitmask( i )->getX();
        int y = this->bitmaskArray->getIthBitmask( i )->getY();
        bool state = this->bitmaskArray->getIthBitmask( i )->getState();
        file << "x=" << setw( 10 ) << x
             << ", y=" << setw( 10 ) << y
             << ", state=" << setw( 1 ) << state
             << std::endl;
    }
}

template< int LogX,
          int LogY >
tnlLeafNode< LogX, LogY >::~tnlLeafNode()
{
    delete this->bitmaskArray;
}

#endif // _TNLLEAFNODE_IMPL_H_INCLUDED_
