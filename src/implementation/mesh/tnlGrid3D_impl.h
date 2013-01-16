/***************************************************************************
                          tnlGrid3D_impl.h  -  description
                             -------------------
    begin                : Jan 16, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLGRID3D_IMPL_H_
#define TNLGRID3D_IMPL_H_

#include <core/tnlAssert.h>

template< typename Real,
          typename Device,
          typename Index >
tnlGrid< 3, Real, Device, Index> :: tnlGrid()
: dofs( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlGrid< 3, Real, Device, Index> :: getTypeStatic()
{
   return tnlString( "tnlGrid< " ) +
          tnlString( Dimensions ) + ", " +
          tnlString( getParameterType< RealType >() ) + ", " +
          tnlString( Device :: getDeviceType() ) + ", " +
          tnlString( getParameterType< IndexType >() ) + " >";
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 3, Real, Device, Index> :: setDimensions( const Index zSize, const Index ySize, const Index xSize )
{
   tnlAssert( xSize > 1,
              cerr << "The number of nodes along x-axis must be larger than 1." );
   tnlAssert( ySize > 1,
              cerr << "The number of nodes along y-axis must be larger than 1." );
   tnlAssert( zSize > 1,
              cerr << "The number of nodes along z-axis must be larger than 1." );

   this -> dimensions. x() = xSize;
   this -> dimensions. y() = ySize;
   this -> dimensions. z() = zSize;
   dofs = zSize * ySize * xSize;
}

template< typename Real,
          typename Device,
          typename Index >
const tnlTuple< 3, Index >& tnlGrid< 3, Real, Device, Index> :: getDimensions() const
{
   return this -> dimensions;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 3, Real, Device, Index> :: setLowerCorner( const tnlTuple< 3, Real >& lowerCorner )
{
   this -> lowerCorner = lowerCorner;
}

template< typename Real,
          typename Device,
          typename Index >
const tnlTuple< 3, Real >& tnlGrid< 3, Real, Device, Index> :: getLowerCorner() const
{
   return this -> lowerCorner;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 3, Real, Device, Index> :: setUpperCorner( const tnlTuple< 3, Real >& upperCorner )
{
   this -> upperCorner = upperCorner;
}

template< typename Real,
          typename Device,
          typename Index >
const tnlTuple< 3, Real >& tnlGrid< 3, Real, Device, Index> :: getUpperCorner() const
{
   return this -> upperCorner;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 3, Real, Device, Index> :: setSpaceStep( const tnlTuple< 3, Real >& spaceStep )
{
   this -> upperCorner. x() = this -> lowerCorner. x() +
                              this -> dimensions. x() *
                              spaceStep. x();
   this -> upperCorner. y() = this -> lowerCorner. y() +
                              this -> dimensions. y() *
                              spaceStep. y();
   this -> upperCorner. z() = this -> lowerCorner. z() +
                              this -> dimensions. z() *
                              spaceStep. z();
}

template< typename Real,
          typename Device,
          typename Index >
tnlTuple< 3, Real > tnlGrid< 3, Real, Device, Index> :: getSpaceStep() const
{
   tnlAssert( dimensions. x() > 0,
              cerr << "Cannot get the space step hx since number of nodes along the x axis is not known in tnlGrid "
                   << this -> getName() );
   tnlAssert( dimensions. y() > 0,
              cerr << "Cannot get the space step hy since number of nodes along the y axis is not known in tnlGrid "
                   << this -> getName() );
   tnlAssert( dimensions. z() > 0,
              cerr << "Cannot get the space step hz since number of nodes along the z axis is not known in tnlGrid "
                   << this -> getName() );

   tnlTuple< 3, RealType > spaceStep;
   spaceStep. x() =
            ( this -> upperCorner. x() - this -> lowerCorner. x() ) /
            ( Real ) ( this -> dimensions. x() - 1 );
   spaceStep. y() =
            ( this -> upperCorner. y() - this -> lowerCorner. y() ) /
            ( Real ) ( this -> dimensions. y() - 1 );
   spaceStep. z() =
            ( this -> upperCorner. z() - this -> lowerCorner. z() ) /
            ( Real ) ( this -> dimensions. z() - 1 );

   return spaceStep;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlGrid< 3, Real, Device, Index> :: getNodeIndex( const Index k, const Index j, const Index i ) const
{
   tnlAssert( i < dimensions. x(),
              cerr << "Index i ( " << i
                   << " ) is out of range ( " << dimensions. x()
                   << " ) in tnlGrid " << this -> getName(); )
   tnlAssert( j < dimensions. y(),
              cerr << "Index j ( " << j
                   << " ) is out of range ( " << dimensions. y()
                   << " ) in tnlGrid " << this -> getName(); )
   tnlAssert( k < dimensions. z(),
            cerr << "Index k ( " << k
                 << " ) is out of range ( " << dimensions. z()
                 << " ) in tnlGrid " << this -> getName(); )

   return ( k * this -> dimensions. y() + j ) * this -> dimensions. x() + i;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlGrid< 3, Real, Device, Index> :: getDofs() const
{
   return this -> dofs;
};


#endif /* TNLGRID3D_IMPL_H_ */