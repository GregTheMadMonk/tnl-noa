/***************************************************************************
                          tnlCoordinates.h  -  description
                             -------------------
    begin                : Feb 27, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLCOORDINATES_H_
#define TNLCOORDINATES_H_

#include <core/vectors/tnlStaticVector.h>

/*!***
 * tnlCoordinates are used mainly by the tnlCommunicator.
 * It is usually meant as coordinate of a node in a finite grid.
 */
template< int Dimensions, typename Index >
class tnlCoordinates : public tnlStaticVector< Dimensions, Index >
{
};

template< typname Index >
class tnlCoordinates< 1, Index > : public tnlStaticVector< 1, Index >
{
   Index getGridSize() const;

   void nextNode( const tnlCoordinates< 1, Index >& dimensions );
};

template< typname Index >
class tnlCoordinates< 2, Index > : public tnlStaticVector< 1, Index >
{
   Index getGridSize() const;

   void nextNode( const tnlCoordinates< 2, Index >& dimensions );
};

template< typname Index >
class tnlCoordinates< 3, Index > : public tnlStaticVector< 1, Index >
{
   Index getGridSize() const;

   void nextNode( const tnlCoordinates< 3, Index >& dimensions );
};

template< typname Index >
Index tnlCoordinates< 1, Index > :: getGridSize() const
{
   return ( *this )[ 0 ];
}

template< typname Index >
void tnlCoordinates< 1, Index > :: nextNode( const tnlCoordinates< 1, Index >& dimensions )
{
   ( *this )[ 0 ] ++;
   ( *this )[ 0 ] = ( *this )[ 0 ] % dimensions[ 0 ];
}


template< typname Index >
Index tnlCoordinates< 2, Index > :: getGridSize() const
{
   return ( *this )[ 0 ] * ( *this )[ 1 ];
}

template< typname Index >
void tnlCoordinates< 2, Index > :: nextNode( const tnlCoordinates< 2, Index >& dimensions )
{

}


template< typname Index >
Index tnlCoordinates< 3, Index > :: getGridSize() const
{
   return ( *this )[ 0 ] * ( *this )[ 1 ] * ( *this )[ 2 ];
}

template< typname Index >
void tnlCoordinates< 3, Index > :: nextNode( const tnlCoordinates< 3, Index >& dimensions )
{

}


#endif /* TNLCOORDINATES_H_ */
