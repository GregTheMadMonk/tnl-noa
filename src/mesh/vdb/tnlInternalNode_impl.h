#ifndef _TNLINTERNALNODE_IMPL_H_INCLUDED_
#define _TNLINTERNALNODE_IMPL_H_INCLUDED_

#include <iostream>
#include <iomanip>
#include "tnlInternalNode.h"
#include "tnlLeafNode.h"
#include "tnlArea2D.h"
#include "tnlCircle2D.h"
#include "tnlBitmask.h"
#include "tnlVDBMath.h"
#include "tnlIterator2D.h"
#include "tnlBitmaskArray.h"

template< int LogX,
          int LogY >
tnlInternalNode< LogX, LogY >::tnlInternalNode( tnlArea2D* area,
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
void tnlInternalNode< LogX, LogY >::setNode( int splitX,
                                             int splitY,
                                             int depth )
{
    int depthX = splitX * tnlVDBMath::power( LogX, this->level - 1 );
    int depthY = splitY * tnlVDBMath::power( LogY, this->level - 1 );
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
    iter->computeBitmaskArray( this->bitmaskArray, this->circle, this->X, this->Y );
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
void tnlInternalNode< LogX, LogY >::print( int splitX,
                                           int splitY,
                                           int depth,
                                           fstream& file )
{
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
    iter->dumpIntoFile( this->bitmaskArray, file, this->level );
    for( int i = 0; i < LogX * LogY; i++ )
        if( this->children[ i ] == NULL )
            continue;
        else
            this->children[ i ]->print( splitX, splitY, depth, file );
}


#endif // _TNLINTERNALNODE_IMPL_H_INCLUDED_
