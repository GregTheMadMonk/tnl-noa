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

/****
 * Identical geometry for 1D
 */

template< typename Real,
          typename Device,
          typename Index >
void tnlIdenticalGridGeometry< 1, Real, Device, Index > :: setParametricStep( const VertexType& parametricStep )
{
   this -> parametricStep = parametricStep;
   this -> elementMeasure = this -> parametricStep. x();
}

template< typename Real,
          typename Device,
          typename Index >
const typename tnlIdenticalGridGeometry< 1, Real, Device, Index > :: VertexType& 
   tnlIdenticalGridGeometry< 1, Real, Device, Index > :: getParametricStep() const
{
   return this -> parametricStep;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlIdenticalGridGeometry< 1, Real, Device, Index > :: getElementCenter( const VertexType& origin,
                                                                             const CoordinatesType& coordinates,
                                                                             VertexType& center ) const
{
   center. x() = ( coordinates. x() + 0.5 ) * parametricStep. x();
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlIdenticalGridGeometry< 1, Real, Device, Index > :: getElementMeasure( const CoordinatesType& i ) const
{
   return elementMeasure;
}

template< typename Real,
          typename Device,
          typename Index >
   template< Index dx >
Real tnlIdenticalGridGeometry< 1, Real, Device, Index > :: getElementsDistance( const Index i ) const
{
   return dx * parametricStep. x();
}

template< typename Real,
          typename Device,
          typename Index >
template< Index dx >
void tnlIdenticalGridGeometry< 1, Real, Device, Index > :: getEdgeCoordinates( const Index i,
                                                                               const VertexType& origin,
                                                                               VertexType& coordinates ) const
{
   coordinates. x() = origin. x() + ( i + 0.5 * ( 1.0 + dx ) ) * parametricStep. x();
}

template< typename Real,
          typename Device,
          typename Index >
template< Index dx >
Real tnlIdenticalGridGeometry< 1, Real, Device, Index > :: getEdgeLength( const Index i ) const
{
   return 0.0;
}

template< typename Real,
          typename Device,
          typename Index >
template< Index dx >
void tnlIdenticalGridGeometry< 1, Real, Device, Index > :: getEdgeNormal( const Index i,
                                                                          VertexType& normal ) const
{
   tnlAssert( dx == 1 || dx == -1, cerr << " dx = " << dx << endl );
   normal. x() = dx;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlIdenticalGridGeometry< 1, Real, Device, Index > :: save( tnlFile& file ) const
{
   if( ! this -> parametricStep. save( file ) )
      return false;
   return true;
};

template< typename Real,
          typename Device,
          typename Index >
bool tnlIdenticalGridGeometry< 1, Real, Device, Index > :: load( tnlFile& file )
{
   if( ! this -> parametricStep. load( file ) )
      return false;
   this -> elementMeasure = this -> parametricStep. x();
   return true;
};

/****
 * Identical geometry for 2D
 */

template< typename Real,
          typename Device,
          typename Index >
void tnlIdenticalGridGeometry< 2, Real, Device, Index > :: setParametricStep( const VertexType& parametricStep )
{
   this -> parametricStep = parametricStep;
   this -> elementMeasure = this -> parametricStep. x() * this -> parametricStep. y();
}

template< typename Real,
          typename Device,
          typename Index >
const typename tnlIdenticalGridGeometry< 2, Real, Device, Index > :: VertexType&
   tnlIdenticalGridGeometry< 2, Real, Device, Index > :: getParametricStep() const
{
   return this -> parametricStep;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlIdenticalGridGeometry< 2, Real, Device, Index > :: getElementCenter( const VertexType& origin,
                                                                             const CoordinatesType& coordinates,
                                                                             VertexType& center ) const
{
   center. x() = ( coordinates. x() + 0.5 ) * parametricStep. x();
   center. y() = ( coordinates. y() + 0.5 ) * parametricStep. y();
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlIdenticalGridGeometry< 2, Real, Device, Index > :: getElementMeasure( const CoordinatesType& coordinates ) const
{
   return elementMeasure;
}

template< typename Real,
          typename Device,
          typename Index >
   template< int dx, int dy >
Real tnlIdenticalGridGeometry< 2, Real, Device, Index > :: getElementCoVolumeMeasure( const CoordinatesType& coordinates ) const
{
   return 0.5 * elementMeasure;
}


template< typename Real,
          typename Device,
          typename Index >
Real tnlIdenticalGridGeometry< 2, Real, Device, Index > :: getElementsDistance( const CoordinatesType& c1,
                                                                                const CoordinatesType& c2 ) const
{
   CoordinatesType c( c2 );
   c -= c1;
   if( c. y() == 0 )
      return parametricStep. x() * fabs( c. x() );
   if( c. x() == 0 )
      return parametricStep. y() * fabs( c. y() );
   return sqrt( c. x() * c. x() + c. y() * c. y() );
}

/*
template< typename Real,
          typename Device,
          typename Index >
template< Index dx, Index dy >
void tnlIdenticalGridGeometry< 2, Real, Device, Index > :: getEdgeCoordinates( const Index i,
                                                                               const Index j,
                                                                               const VertexType& origin,
                                                                               VertexType& coordinates ) const
{
   coordinates. x() = origin. x() + ( i + 0.5 * ( 1.0 + dx ) ) * parametricStep. x();
   coordinates. y() = origin. y() + ( j + 0.5 * ( 1.0 + dy ) ) * parametricStep. y();
}

template< typename Real,
          typename Device,
          typename Index >
template< Index dx, Index dy >
Real tnlIdenticalGridGeometry< 2, Real, Device, Index > :: getEdgeLength( const Index i,
                                                                          const Index j ) const
{
   if( dy == 0 && dx == 1 )
      return parametricStep. y();
   if( dy == 1 && dx == 0 )
      return parametricStep. x();
   tnlAssert( false, cerr << "Bad values of dx and dy - dx = " << dx << " dy = " << dy );
}*/

template< typename Real,
          typename Device,
          typename Index >
template< Index dx, Index dy >
void tnlIdenticalGridGeometry< 2, Real, Device, Index > :: getEdgeNormal( const CoordinatesType& coordinates,                                                                     
                                                                          VertexType& normal ) const
{
   tnlAssert( ( dx == 0 || dx == 1 || dx == -1 ||
                dy == 0 || dy == 1 || dy == -1 ) &&
               dx * dy == 0, cerr << " dx = " << dx << " dy = " << dy << endl );
   normal. x() = dx * parametricStep. y();
   normal. y() = dy * parametricStep. x();
}

template< typename Real,
          typename Device,
          typename Index >
   template< Index dx, Index dy >
void tnlIdenticalGridGeometry< 2, Real, Device, Index > :: getVertex( const CoordinatesType& coordinates,
                                                                      const VertexType& origin,
                                                                      VertexType& vertex ) const
{
   tnlAssert( ( dx == 0 || dx == 1 || dx == -1 ||
                dy == 0 || dy == 1 || dy == -1 ), cerr << " dx = " << dx << " dy = " << dy << endl );
   vertex. x() = origin. x() + ( coordinates. x() + 0.5 * ( 1 + dx ) ) * parametricStep. x();
   vertex. y() = origin. y() + ( coordinates. y() + 0.5 * ( 1 + dy ) ) * parametricStep. y();
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlIdenticalGridGeometry< 2, Real, Device, Index > :: save( tnlFile& file ) const
{
   if( ! this -> parametricStep. save( file ) )
      return false;
   return true;
};

template< typename Real,
          typename Device,
          typename Index >
bool tnlIdenticalGridGeometry< 2, Real, Device, Index > :: load( tnlFile& file )
{
   if( ! this -> parametricStep. load( file ) )
      return false;
   this -> elementMeasure = this -> parametricStep. x() * this -> parametricStep. y();
   return true;
};

#endif /* TNLIDENTICALGRIDGEOMETRY_IMPL_H_ */
