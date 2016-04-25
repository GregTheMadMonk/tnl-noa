#ifndef _TNLINTERNALNODE_IMPL_H_INCLUDED_
#define _TNLINTERNALNODE_IMPL_H_INCLUDED_

#include <iostream>
#include <iomanip>
#include "tnlInternalNode.h"
#include "tnlLeafNode.h"

template< int LogX,
          int LogY >
tnlInternalNode< LogX, LogY >::tnlInternalNode( tnlArea2D* area,
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
void tnlInternalNode< LogX, LogY >::setNode( int splitX,
                                             int splitY,
                                             int depth )
{
    tnlNode< LogX, LogY >::setNode( splitX, splitY, this->bitmaskArray );
    this->setChildren( splitX, splitY, depth );
}

template< int LogX,
          int LogY >
void tnlInternalNode< LogX, LogY >::setChildren( int splitX,
                                                 int splitY,
                                                 int depth )
{
    for( int i = 0; i < LogY; i++ )
        for( int j = 0; j < LogX; j++ )
        {
            int index = i * LogY + j;
            if( !this->bitmaskArray->getIthBitmask( index )->getState() )
                this->children[ index ] = NULL;
            else if( this->level < depth - 1 )
            {
                //std::cout << "creating new node, level = " << this->level << std::endl;
                int X = this->X * LogX + j;
                int Y = this->Y * LogY + i;
                this->children[ index ] = new tnlInternalNode< LogX, LogY >( this->area,
                                                                             this->circle,
                                                                             X,
                                                                             Y,
                                                                             this->level + 1 );
                this->children[ index ]->setNode( splitX, splitY, depth );
            }
            else
            {
                int X = this->X * LogX + j;
                int Y = this->Y * LogY + i;
                this->children[ index ] = new tnlLeafNode< LogX, LogY >( this->area,
                                                                         this->circle,
                                                                         X,
                                                                         Y,
                                                                         this->level + 1 );
                this->children[ index ]->setNode( splitX, splitY, depth );
            }    
        }
}

template< int LogX,
          int LogY >
void tnlInternalNode< LogX, LogY >::write( fstream& file,
                                           int level )
{
    for( int i = 0; i < LogX * LogY; i++ )
    {
        if( this->level == level )
        {
            int x = this->bitmaskArray->getIthBitmask( i )->getX();
            int y = this->bitmaskArray->getIthBitmask( i )->getY();
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

template< int LogX,
          int LogY >
tnlInternalNode< LogX, LogY >::~tnlInternalNode()
{
    delete this->bitmaskArray;
    for( int i = 0; i < LogX * LogY; i++ )
        delete this->children[ i ];
    delete [] this->children;
}


#endif // _TNLINTERNALNODE_IMPL_H_INCLUDED_
