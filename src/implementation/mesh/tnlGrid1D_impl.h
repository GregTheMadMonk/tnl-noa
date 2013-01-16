/***************************************************************************
                          tnlGrid1D_impl.h  -  description
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

#ifndef TNLGRID1D_IMPL_H_
#define TNLGRID1D_IMPL_H_

#include <core/tnlString.h>
#include <core/tnlAssert.h>

template< typename Real,
          typename Device,
          typename Index >
tnlGrid< 1, Real, Device, Index> :: tnlGrid()
: dofs( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlGrid< 1, Real, Device, Index> :: getTypeStatic()
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
void tnlGrid< 1, Real, Device, Index> :: setDimensions( const Index xSize )
{
   tnlAssert( xSize > 1,
              cerr << "The number of nodes along x-axis must be larger than 1." );
   this -> dimensions. x() = xSize;
   dofs = xSize;
}

template< typename Real,
          typename Device,
          typename Index >
const tnlTuple< 1, Index >& tnlGrid< 1, Real, Device, Index> :: getDimensions() const
{
   return this -> dimensions;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 1, Real, Device, Index> :: setLowerCorner( const tnlTuple< 1, Real >& lowerCorner )
{
   this -> lowerCorner = lowerCorner;
}

template< typename Real,
          typename Device,
          typename Index >
const tnlTuple< 1, Real >& tnlGrid< 1, Real, Device, Index> :: getLowerCorner() const
{
   return this -> lowerCorner;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 1, Real, Device, Index> :: setUpperCorner( const tnlTuple< 1, Real >& upperCorner )
{
   this -> upperCorner = upperCorner;
}

template< typename Real,
          typename Device,
          typename Index >
const tnlTuple< 1, Real >& tnlGrid< 1, Real, Device, Index> :: getUpperCorner() const
{
   return this -> upperCorner;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 1, Real, Device, Index> :: setSpaceStep( const tnlTuple< 1, Real >& spaceStep )
{
   this -> upperCorner. x() = this -> lowerCorner. x() +
                              this -> dimensions. x() *
                              spaceStep. x();
}

template< typename Real,
          typename Device,
          typename Index >
tnlTuple< 1, Real > tnlGrid< 1, Real, Device, Index> :: getSpaceStep() const
{
   tnlAssert( dimensions. x() > 0,
              cerr << "Cannot get the space step hx since number of nodes along the x axis is not known in tnlGrid "
                   << this -> getName() );
   tnlTuple< 1, RealType > spaceStep;
   spaceStep. x() =
            ( this -> upperCorner. x() - this -> lowerCorner. x() ) /
            ( Real ) ( this -> dimensions. x() - 1 );
   return spaceStep;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlGrid< 1, Real, Device, Index> :: getNodeIndex( const Index i ) const
{
   tnlAssert( i < dimensions. x(),
              cerr << "Index i ( " << i
                   << " ) is out of range ( " << dimensions. x()
                   << " ) in tnlGrid " << this -> getName(); )
   return i;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlGrid< 1, Real, Device, Index> :: getDofs() const
{
   return this -> dofs;
};


#endif /* TNLGRID1D_IMPL_H_ */