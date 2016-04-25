#ifndef _TNLROOTNODE_IMPL_H_INCLUDED_
#define _TNLROOTNODE_IMPL_H_INCLUDED_

#include <iostream>
#include <iomanip>
#include <string>
#include "tnlNode.h"
#include "tnlRootNode.h"
#include "tnlInternalNode.h"
#include "tnlLeafNode.h"


template< unsigned size,
          int LogX,
          int LogY >
tnlRootNode< size, LogX, LogY >::tnlRootNode( tnlArea2D* area,
                                              tnlCircle2D* circle,
                                              unsigned nodesX,
                                              unsigned nodesY,
                                              unsigned depth )
: tnlNode< LogX, LogY >::tnlNode( area, circle, 0, 0, 0 )
{
    this->nodesX = nodesX;
    this->nodesY = nodesY;
    this->bitmaskArray = new tnlBitmaskArray< size >();
    this->depth = depth;
}

template< unsigned size, 
          int LogX,
          int LogY >
void tnlRootNode< size, LogX, LogY >::setNode()
{
    float stepX = ( this->area->getEndX() - this->area->getStartX() ) / this->nodesX;
    float stepY = ( this->area->getEndY() - this->area->getStartY() ) / this->nodesY;
    float startX = this->area->getStartX();
    float startY = this->area->getStartY();
    for( int i = 0; i < this->nodesX; i++ )
        for( int j = 0; j < this->nodesY; j++ )
        {
            float x1 = startX + j * stepX;
            float x2 = startX + ( j + 1 ) * stepX;
            float y1 = startY + i * stepY;
            float y2 = startY + ( i + 1 ) * stepY;
            bool state = this->circle->isIntercept( x1, x2, y1, y2 );
            int X = j;
            int Y = i;
            tnlBitmask* bitmask = new tnlBitmask( state, X, Y );
            this->bitmaskArray->setIthBitmask( i * this->nodesX + j, bitmask);
        }
}

template< unsigned size,
          int LogX,
          int LogY >
void tnlRootNode< size, LogX, LogY >::createTree()
{
    this->setNode(); // first we need to create root node
    for( int i = 0; i < this->nodesY; i++ )
        for( int j = 0; j < this-> nodesX; j++ )
        {
            int index = i * this->nodesY + j;
            if( !this->bitmaskArray->getIthBitmask( index )->getState() )
                this->children[ index ] = NULL;
            else if( this->level < this->depth - 1 )
            {
                int X = j;
                int Y = i;
                this->children[ index ] = new tnlInternalNode< LogX, LogY >( this->area,
                                                                             this->circle,
                                                                             X,
                                                                             Y,
                                                                             this->level + 1 );
                this->children[ index ]->setNode( nodesX, nodesY, this->depth );
            }
            else
            {
                int X = j;
                int Y = i;
                this->children[ index ] = new tnlLeafNode< LogX, LogY >( this->area,
                                                                         this->circle,
                                                                         X,
                                                                         Y,
                                                                         this->level + 1 );
                this->children[ index ]->setNode( nodesX, nodesY, this->depth );
            }
        }
}

template< unsigned size,
          int LogX,
          int LogY >
void tnlRootNode< size, LogX, LogY >::printStates( fstream& file )
{
    float stepX = ( this->area->getEndX() - this->area->getStartX() ) / this->nodesX;
    float stepY = ( this->area->getEndY() - this->area->getStartY() ) / this->nodesY;
    tnlIterator2D< size, LogX, LogY >* iter = new tnlIterator2D< size, LogX, LogY >( this->nodesX,
                                                                                     this->nodesY,
                                                                                     stepX,
                                                                                     stepY,
                                                                                     this->area->getStartX(),
                                                                                     this->area->getStartY() );
    iter->dumpIntoFile( this->bitmaskArray, file );

    for( int i = 0; i < size; i++ )
        if( this->children[ i ] == NULL )
            continue;
        else
            children[ i ]->print( nodesX, nodesY, this->depth, file );
}

template< unsigned size,
          int LogX,
          int LogY >
void tnlRootNode< size, LogX, LogY >::write()
{
    for( int i = 0; i < this->depth; i++ )
    {
        std::string filename = "nodesLevel_" + std::to_string( i );
        fstream f;
        f.open( filename, ios::out | ios::trunc );
        int startX = this->area->getStartX();
        int endX = this->area->getEndX();
        int startY = this->area->getStartY();
        int endY = this->area->getEndY();
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
        for( int j = 0; j < size; j++ )
        {
            if( this->level == i )
            {
                int x = this->bitmaskArray->getIthBitmask( j )->getX();
                int y = this->bitmaskArray->getIthBitmask( j )->getY();
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

template< unsigned size,
          int LogX,
          int LogY >
tnlRootNode< size, LogX, LogY >::~tnlRootNode()
{
    delete this->bitmaskArray;
    for( int i = 0; i < size; i++ ) 
        delete this->children[ i ];
    delete [] this->children;
}

#endif // _TNLROOTNODE_IMPL_H_INCLUDED_
