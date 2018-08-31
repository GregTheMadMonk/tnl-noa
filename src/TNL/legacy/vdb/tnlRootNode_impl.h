#ifndef _TNLROOTNODE_IMPL_H_INCLUDED_
#define _TNLROOTNODE_IMPL_H_INCLUDED_

#include <iostream>
#include <iomanip>
#include <string>
#include "tnlNode.h"
#include "tnlRootNode.h"
#include "tnlInternalNode.h"
#include "tnlLeafNode.h"


template< typename Real,
          typename Index,
          unsigned Size,
          Index LogX,
          Index LogY >
tnlRootNode< Real, Index, Size, LogX, LogY >::tnlRootNode( tnlArea2D< Real >* area,
                                                           tnlCircle2D< Real >* circle,
                                                           unsigned nodesX,
                                                           unsigned nodesY,
                                                           unsigned depth )
: tnlNode< Real, Index, LogX, LogY >::tnlNode( area, circle, 0, 0, 0 )
{
    this->nodesX = nodesX;
    this->nodesY = nodesY;
    this->bitmaskArray = new tnlBitmaskArray< Size >();
    this->depth = depth;
}

template< typename Real,
          typename Index,
          unsigned Size, 
          Index LogX,
          Index LogY >
void tnlRootNode< Real, Index, Size, LogX, LogY >::setNode()
{
    Real stepX = ( this->area->getEndX() - this->area->getStartX() ) / this->nodesX;
    Real stepY = ( this->area->getEndY() - this->area->getStartY() ) / this->nodesY;
    Real startX = this->area->getStartX();
    Real startY = this->area->getStartY();
    for( Index i = 0; i < this->nodesX; i++ )
        for( Index j = 0; j < this->nodesY; j++ )
        {
            Real x1 = startX + j * stepX;
            Real x2 = startX + ( j + 1 ) * stepX;
            Real y1 = startY + i * stepY;
            Real y2 = startY + ( i + 1 ) * stepY;
            bool state = this->circle->isIntercept( x1, x2, y1, y2 );
            Index X = j;
            Index Y = i;
            tnlBitmask* bitmask = new tnlBitmask( state, X, Y );
            this->bitmaskArray->setIthBitmask( i * this->nodesX + j, bitmask);
        }
}

template< typename Real,
          typename Index,
          unsigned Size,
          Index LogX,
          Index LogY >
void tnlRootNode< Real, Index, Size, LogX, LogY >::createTree()
{
    this->setNode(); // first we need to create root node
    for( Index i = 0; i < this->nodesY; i++ )
        for( Index j = 0; j < this-> nodesX; j++ )
        {
            Index index = i * this->nodesY + j;
            if( !this->bitmaskArray->getIthBitmask( index )->getState() )
                this->children[ index ] = NULL;
            else if( this->level < this->depth - 1 )
            {
                Index X = j;
                Index Y = i;
                this->children[ index ] = new tnlInternalNode< Real, Index, LogX, LogY >( this->area,
                                                                                          this->circle,
                                                                                          X,
                                                                                          Y,
                                                                                          this->level + 1 );
                this->children[ index ]->setNode( nodesX, nodesY, this->depth );
            }
            else
            {
                Index X = j;
                Index Y = i;
                this->children[ index ] = new tnlLeafNode< Real, Index, LogX, LogY >( this->area,
                                                                                      this->circle,
                                                                                      X,
                                                                                      Y,
                                                                                      this->level + 1 );
                this->children[ index ]->setNode( nodesX, nodesY, this->depth );
            }
        }
}

template< typename Real,
          typename Index,
          unsigned Size,
          Index LogX,
          Index LogY >
void tnlRootNode< Real, Index, Size, LogX, LogY >::write()
{
    for( Index i = 0; i < this->depth; i++ )
    {
        std::string filename = "nodesLevel_" + std::to_string( i );
        fstream f;
        f.open( filename, ios::out | ios::trunc );
        Index startX = this->area->getStartX();
        Index endX = this->area->getEndX();
        Index startY = this->area->getStartY();
        Index endY = this->area->getEndY();
        f << "startx=" << setw( 10 ) << startX
          << ", endx=" << setw( 10 ) << endX
          << ", starty=" <<setw( 10 ) << startY
          << ", endy=" << setw( 10 ) << endY
          << ", level=" << setw( 10 ) << i
          << std::endl;
        f << "rootSplitX=" << setw( 10 ) << this->nodesX
          << ", rootSplitY=" << setw( 10 ) << this->nodesY
          << ", LogX=" << setw( 10 ) << LogX
          << ", LogY=" << setw( 10 ) << LogY 
          << std::endl << std::endl;
        for( Index j = 0; j < Size; j++ )
        {
            if( this->level == i )
            {
                Index x = this->bitmaskArray->getIthBitmask( j )->getX();
                Index y = this->bitmaskArray->getIthBitmask( j )->getY();
                bool state = this->bitmaskArray->getIthBitmask( j )->getState();
                f << "x=" << setw( 10 ) << x
                  << ", y=" << setw( 10 ) << y
                  << ", state=" << setw( 1 ) << state
                  << std::endl;
            }
            else if( this->children[ j ] )
                this->children[ j ]->write( f, i );
        }
    }
}

template< typename Real,
          typename Index,
          unsigned Size,
          Index LogX,
          Index LogY >
tnlRootNode< Real, Index, Size, LogX, LogY >::~tnlRootNode()
{
    delete this->bitmaskArray;
    for( Index i = 0; i < Size; i++ ) 
        delete this->children[ i ];
    delete [] this->children;
}

#endif // _TNLROOTNODE_IMPL_H_INCLUDED_
