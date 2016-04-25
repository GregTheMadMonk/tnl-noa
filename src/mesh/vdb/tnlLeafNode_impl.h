#ifndef _TNLLEAFNODE_IMPL_H_INCLUDED_
#define _TNLLEAFNODE_IMPL_H_INCLUDED_

#include "tnlLeafNode.h"
#include <iostream>
#include <iomanip>

template< typename Real,
          typename Index,
          Index LogX,
          Index LogY >
tnlLeafNode< Real, Index, LogX, LogY >::tnlLeafNode( tnlArea2D< Real >* area,
                                                     tnlCircle2D< Real >* circle,
                                                     Index X,
                                                     Index Y,
                                                     Index level )
: tnlNode< Real, Index, LogX, LogY >::tnlNode( area, circle, X, Y, level )
{
    this->bitmaskArray = new tnlBitmaskArray< LogX * LogY >();
}

template< typename Real,
          typename Index,
          Index LogX,
          Index LogY >
void tnlLeafNode< Real, Index, LogX, LogY >::setNode( Index splitX,
                                                      Index splitY,
                                                      Index depth )
{
    tnlNode< Real, Index, LogX, LogY >::setNode( splitX, splitY, this->bitmaskArray );
}

template< typename Real,
          typename Index,
          Index LogX,
          Index LogY >
void tnlLeafNode< Real, Index, LogX, LogY >::write( fstream& file,
                                                    Index level )
{
    for( Index i = 0; i < LogX * LogY; i++ )
    {
        Index x = this->bitmaskArray->getIthBitmask( i )->getX();
        Index y = this->bitmaskArray->getIthBitmask( i )->getY();
        bool state = this->bitmaskArray->getIthBitmask( i )->getState();
        file << "x=" << setw( 10 ) << x
             << ", y=" << setw( 10 ) << y
             << ", state=" << setw( 1 ) << state
             << std::endl;
    }
}

template< typename Real,
          typename Index,
          Index LogX,
          Index LogY >
tnlLeafNode< Real, Index, LogX, LogY >::~tnlLeafNode()
{
    delete this->bitmaskArray;
}

#endif // _TNLLEAFNODE_IMPL_H_INCLUDED_
