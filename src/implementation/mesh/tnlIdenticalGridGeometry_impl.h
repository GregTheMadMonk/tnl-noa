/***************************************************************************
                          tnlIdenticalGridGeometry_impl.h  -  description
                             -------------------
    begin                : May 1, 2013
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

#ifndef TNLIDENTICALGRIDGEOMETRY_IMPL_H_
#define TNLIDENTICALGRIDGEOMETRY_IMPL_H_

#include <core/tnlFile.h>
#include <core/tnlAssert.h>

template< typename Real,
          typename Device,
          typename Index >
void tnlIdenticalGridGeometry< 2, Real, Device, Index > :: setParametricStep( const tnlTuple< 2, Real >& parametricStep )
{
   this -> parametricStep = parametricStep;
   this -> elementMeasure - this -> parametricStep. x() * this -> parametricStep. y();
}

template< typename Real,
          typename Device,
          typename Index >
void tnlIdenticalGridGeometry< 2, Real, Device, Index > :: getElementCoordinates( const Index j,
                                                                                  const Index i,
                                                                                  tnlTuple< 2, Real >& coordinates ) const
{
   coordinates. x() = i * parametricStep. x();
   coordinates. y() = j * parametricStep. y();
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlIdenticalGridGeometry< 2, Real, Device, Index > :: getElementMeasure( const Index j,
                                                                              const Index i ) const
{
   return elementMeasure;
}

template< typename Real,
          typename Device,
          typename Index >
   template< Index dy, Index dx >
Real tnlIdenticalGridGeometry< 2, Real, Device, Index > :: getElementsDistance( const Index j,
                                                                                const Index i ) const
{
   if( dy == 0 && dx == 1 )
      return parametricStep. x();
   if( dy == 1 && dx == 0 )
      return parametricStep. y();
   const Real x = dx * parametricStep. x();
   const Real y = dy * parametricStep. y();
   return sqrt( dx * dx + dy * dy );
}

template< typename Real,
          typename Device,
          typename Index >
template< Index dy, Index dx >
void tnlIdenticalGridGeometry< 2, Real, Device, Index > :: getEdgeCoordinates( const Index j,
                                                                               const Index i,
                                                                               tnlTuple< 2, Real >& coordinates ) const
{
   coordinates. x() = origin. x() + ( i + 0.5 * ( 1.0 + dx ) ) * parametricStep. x();
   coordinates. y() = origin. y() + ( j + 0.5 * ( 1.0 + dy ) ) * parametricStep. y();
}

template< typename Real,
          typename Device,
          typename Index >
template< int dy, int dx >
Real tnlIdenticalGridGeometry< 2, Real, Device, Index > :: getEdgeLength( const Index j,
                                                                          const Index i ) const
{
   if( dy == 0 && dx == 1 )
      return parametricStep. y();
   if( dy == 1 && dx == 0 )
      return parametricStep. x();
   tnlAssert( false, cerr << "Bad values of dx and dy - dx = " << dx << " dy = " << dy );
}

template< typename Real,
          typename Device,
          typename Index >
template< int dy, int dx >
void tnlIdenticalGridGeometry< 2, Real, Device, Index > :: getEdgeNormal( const Index j,
                                                                          const Index i,
                                                                          tnlTuple< 2, Real >& normal ) const
{
   tnlAssert( ( dx == 0 || dx == 1 || dx == -1 ||
                dy == 0 || dy == 1 || dy == -1 ) &&
               dx * dy == 0, cerr << " dx = " << dx << " dy = " << dy << endl );
   normal. x() = dx;
   normal. y() = dy;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlIdenticalGridGeometry< 2, Real, Device, Index > :: save( tnlFile& file ) const
{
   return true;
};

template< typename Real,
          typename Device,
          typename Index >
bool tnlIdenticalGridGeometry< 2, Real, Device, Index > :: load( tnlFile& file )
{
   return true;
};

#endif /* TNLIDENTICALGRIDGEOMETRY_IMPL_H_ */
