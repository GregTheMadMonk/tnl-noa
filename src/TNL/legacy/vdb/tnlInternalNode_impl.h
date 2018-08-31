#ifndef _TNLINTERNALNODE_IMPL_H_INCLUDED_
#define _TNLINTERNALNODE_IMPL_H_INCLUDED_

#include <iostream>
#include <iomanip>
#include "tnlInternalNode.h"
#include "tnlLeafNode.h"

template< typename Real,
          typename Index,
          Index LogX,
          Index LogY >
tnlInternalNode< Real, Index, LogX, LogY >::tnlInternalNode( tnlArea2D< Real >* area,
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
void tnlInternalNode< Real, Index, LogX, LogY >::setNode( Index splitX,
                                                          Index splitY,
                                                          Index depth )
{
    tnlNode< Real, Index, LogX, LogY >::setNode( splitX, splitY, this->bitmaskArray );
    this->setChildren( splitX, splitY, depth );
}

template< typename Real,
          typename Index,
          Index LogX,
          Index LogY >
void tnlInternalNode< Real, Index, LogX, LogY >::setChildren( Index splitX,
                                                              Index splitY,
                                                              Index depth )
{
    for( Index i = 0; i < LogY; i++ )
        for( Index j = 0; j < LogX; j++ )
        {
            Index index = i * LogY + j;
            if( !this->bitmaskArray->getIthBitmask( index )->getState() )
                this->children[ index ] = NULL;
            else if( this->level < depth - 1 )
            {
                //std::cout << "creating new node, level = " << this->level << std::endl;
                Index X = this->X * LogX + j;
                Index Y = this->Y * LogY + i;
                this->children[ index ] = new tnlInternalNode< Real, Index, LogX, LogY >( this->area,
                                                                             this->circle,
                                                                             X,
                                                                             Y,
                                                                             this->level + 1 );
                this->children[ index ]->setNode( splitX, splitY, depth );
            }
            else
            {
                Index X = this->X * LogX + j;
                Index Y = this->Y * LogY + i;
                this->children[ index ] = new tnlLeafNode< Real, Index, LogX, LogY >( this->area,
                                                                         this->circle,
                                                                         X,
                                                                         Y,
                                                                         this->level + 1 );
                this->children[ index ]->setNode( splitX, splitY, depth );
            }    
        }
}

template< typename Real,
          typename Index,
          Index LogX,
          Index LogY >
void tnlInternalNode< Real, Index, LogX, LogY >::write( fstream& file,
                                                        Index level )
{
    for( Index i = 0; i < LogX * LogY; i++ )
    {
        if( this->level == level )
        {
            Index x = this->bitmaskArray->getIthBitmask( i )->getX();
            Index y = this->bitmaskArray->getIthBitmask( i )->getY();
            bool state = this->bitmaskArray->getIthBitmask( i )->getState();
            file << "x=" << setw( 10 ) << x
                 << ", y=" << setw( 10 ) << y
                 << ", state=" << setw( 1 ) << state
                 << std::endl;
        }
        else if( this->children[ i ] )
            this->children[ i ]->write( file, level );
    }
}

template< typename Real,
          typename Index,
          Index LogX,
          Index LogY >
tnlInternalNode< Real, Index, LogX, LogY >::~tnlInternalNode()
{
    delete this->bitmaskArray;
    for( Index i = 0; i < LogX * LogY; i++ )
        delete this->children[ i ];
    delete [] this->children;
}


#endif // _TNLINTERNALNODE_IMPL_H_INCLUDED_
