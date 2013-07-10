/***************************************************************************
                          tnlLinearGridGeometry_impl.h  -  description
                             -------------------
    begin                : May 10, 2013
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
#ifndef TNLLINEARGRIDGEOMETRY_IMPL_H_
#define TNLLINEARGRIDGEOMETRY_IMPL_H_

#include <core/tnlFile.h>
#include <core/tnlAssert.h>

/****
 * Linear geometry for 1D
 */

template< typename Real,
          typename Device,
          typename Index >
void tnlLinearGridGeometry< 1, Real, Device, Index > :: setParametricStep( const VertexType& parametricStep )
{
   this -> parametricStep = parametricStep;
   this -> elementMeasure = this -> parametricStep. x();
}

template< typename Real,
          typename Device,
          typename Index >
const typename tnlLinearGridGeometry< 1, Real, Device, Index > :: VertexType&
   tnlLinearGridGeometry< 1, Real, Device, Index > :: getParametricStep() const
{
   return this -> parametricStep;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlLinearGridGeometry< 1, Real, Device, Index > :: getElementCenter( const VertexType& origin,
                                                                             const CoordinatesType& coordinates,
                                                                             VertexType& center ) const
{
   center. x() = ( coordinates. x() + 0.5 ) * parametricStep. x();
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlLinearGridGeometry< 1, Real, Device, Index > :: getElementMeasure( const CoordinatesType& i ) const
{
   return elementMeasure;
}

template< typename Real,
          typename Device,
          typename Index >
   template< Index dx >
Real tnlLinearGridGeometry< 1, Real, Device, Index > :: getElementsDistance( const Index i ) const
{
   return dx * parametricStep. x();
}

template< typename Real,
          typename Device,
          typename Index >
template< Index dx >
void tnlLinearGridGeometry< 1, Real, Device, Index > :: getEdgeCoordinates( const Index i,
                                                                               const VertexType& origin,
                                                                               VertexType& coordinates ) const
{
   coordinates. x() = origin. x() + ( i + 0.5 * ( 1.0 + dx ) ) * parametricStep. x();
}

template< typename Real,
          typename Device,
          typename Index >
template< Index dx >
Real tnlLinearGridGeometry< 1, Real, Device, Index > :: getEdgeLength( const Index i ) const
{
   return 0.0;
}

template< typename Real,
          typename Device,
          typename Index >
template< Index dx >
void tnlLinearGridGeometry< 1, Real, Device, Index > :: getEdgeNormal( const Index i,
                                                                          VertexType& normal ) const
{
   tnlAssert( dx == 1 || dx == -1, cerr << " dx = " << dx << endl );
   normal. x() = dx;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlLinearGridGeometry< 1, Real, Device, Index > :: save( tnlFile& file ) const
{
   if( ! this -> parametricStep. save( file ) )
      return false;
   return true;
};

template< typename Real,
          typename Device,
          typename Index >
bool tnlLinearGridGeometry< 1, Real, Device, Index > :: load( tnlFile& file )
{
   if( ! this -> parametricStep. load( file ) )
      return false;
   this -> elementMeasure = this -> parametricStep. x();
   return true;
};

/****
 * Linear geometry for 2D
 */

template< typename Real,
          typename Device,
          typename Index >
tnlLinearGridGeometry< 2, Real, Device, Index > :: tnlLinearGridGeometry()
: numberOfSegments( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlLinearGridGeometry< 2, Real, Device, Index > :: getTypeStatic()
{
   return tnlString( "tnlLinearGridGeometry< 2, " ) +
          getParameterType< RealType >() + ", " +
          Device :: getDeviceType() + ", " +
          getParameterType< IndexType >() + " > ";
}

template< typename Real,
          typename Device,
          typename Index >
void tnlLinearGridGeometry< 2, Real, Device, Index > :: setParametricStep( const VertexType& parametricStep )
{
   this -> parametricStep = parametricStep;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlLinearGridGeometry< 2, Real, Device, Index > :: setProportions( const VertexType& proportions )
{
   this -> proportions = proportions;
}

template< typename Real,
          typename Device,
          typename Index >
const typename tnlLinearGridGeometry< 2, Real, Device, Index > :: VertexType&
   tnlLinearGridGeometry< 2, Real, Device, Index > :: getProportions() const
{
   return this -> proportions;
}

template< typename Real,
          typename Device,
          typename Index >
const typename tnlLinearGridGeometry< 2, Real, Device, Index > :: VertexType&
   tnlLinearGridGeometry< 2, Real, Device, Index > :: getParametricStep() const
{
   return this -> parametricStep;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlLinearGridGeometry< 2, Real, Device, Index > :: getElementCenter( const VertexType& origin,
                                                                          const CoordinatesType& coordinates,
                                                                          VertexType& center ) const
{
   center. x() = ( coordinates. x() + 0.5 ) * parametricStep. x();
   center. y() = ( coordinates. y() + 0.5 ) * parametricStep. y();
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlLinearGridGeometry< 2, Real, Device, Index > :: getElementMeasure( const CoordinatesType& coordinates ) const
{
   VertexType v0, v1, v2, v3;
   const VertexType origin( 0.0 );
   this -> template getVertex< -1, -1 >( coordinates, origin, v0 );
   this -> template getVertex<  1, -1 >( coordinates, origin, v1 );
   this -> template getVertex<  1,  1 >( coordinates, origin, v2 );
   this -> template getVertex< -1,  1 >( coordinates, origin, v3 );

   return tnlTriangleArea( v0, v1, v3 ) + tnlTriangleArea( v2, v3, v1 );

   //return elementMeasure;
}

template< typename Real,
          typename Device,
          typename Index >
   template< int dx, int dy >
Real tnlLinearGridGeometry< 2, Real, Device, Index > :: getDualElementMeasure( const CoordinatesType& coordinates ) const
{
   tnlAssert( ( dx == 0 && ( dy == 1 || dy == -1 ) ) ||
              ( dy == 0 && ( dx == 1 || dx == -1 ) ),
              cerr << " dx = " << dx << " dy = " << dy << endl );
   VertexType v0, v1, v2, v3;
   const VertexType origin( 0.0 );
   this -> getElementCenter( origin, coordinates, v0 );
   if( dy == 0 )
   {
      this -> template getVertex< dx, -1 >( coordinates, origin, v1 );
      this -> template getVertex< dx, 1 >( coordinates, origin, v2 );
      CoordinatesType c2( coordinates );
      c2. x() += dx;
      this -> getElementCenter( origin, c2, v3 );
   }
   if( dx == 0 )
   {
      this -> template getVertex< -1, dy >( coordinates, origin, v1 );
      this -> template getVertex<  1, dy >( coordinates, origin, v2 );
      CoordinatesType c2( coordinates );
      c2. y() += dy;
      this -> getElementCenter( origin, c2, v3 );
   }
   return tnlTriangleArea( v0, v1, v3 ) + tnlTriangleArea( v2, v3, v1 );
}

template< typename Real,
          typename Device,
          typename Index >
template< Index dx, Index dy >
void tnlLinearGridGeometry< 2, Real, Device, Index > :: getEdgeNormal( const CoordinatesType& coordinates,
                                                                       VertexType& normal ) const
{
   tnlAssert( ( dx == 0 && ( dy == 1 || dy == -1 ) ) ||
              ( dy == 0 && ( dx == 1 || dx == -1 ) ),
              cerr << " dx = " << dx << " dy = " << dy << endl );
   VertexType v1, v2, origin( 0.0 );
   if( dy == 0 )
   {
      if( dx == 1 )
      {
         this -> getVertex< 1, 1 >( coordinates, origin, v1 );
         this -> getVertex< 1, -1 >( coordinates, origin, v2 );
      }
      else // dx == -1
      {
         this -> getVertex< -1, -1 >( coordinates, origin, v1 );
         this -> getVertex< -1, 1 >( coordinates, origin, v2 );
      }
   }
   else // dx == 0
   {
      if( dy == 1 )
      {
         this -> getVertex< -1, 1 >( coordinates, origin, v1 );
         this -> getVertex< 1, 1 >( coordinates, origin, v2 );
      }
      else
      {
         this -> getVertex< 1, -1 >( coordinates, origin, v1 );
         this -> getVertex< -1, -1 >( coordinates, origin, v2 );
      }
   }
   normal. x() = v1. y() - v2. y();
   normal. y() = v2. x() - v1. x();
}

template< typename Real,
          typename Device,
          typename Index >
   template< Index dx, Index dy >
void tnlLinearGridGeometry< 2, Real, Device, Index > :: getVertex( const CoordinatesType& coordinates,
                                                                   const VertexType& origin,
                                                                   VertexType& vertex ) const
{
   tnlAssert( ( dx == 0 || dx == 1 || dx == -1 ||
                dy == 0 || dy == 1 || dy == -1 ), cerr << " dx = " << dx << " dy = " << dy << endl );
   const RealType x = ( coordinates. x() + 0.5 * ( 1 + dx ) ) * parametricStep. x();
   const RealType y = ( coordinates. y() + 0.5 * ( 1 + dy ) ) * parametricStep. y();
   if( this -> numberOfSegments == 0  )
   {
      vertex. x() = origin. x() + x;
      vertex. y() = origin. y() + y;
   }
   else
   {
      //y = proportions. y() * pow( y / proportions. y(), 2.0 );
      Index i( 0 );
      while( ySegments[ i ] <= y && i < this -> numberOfSegments - 1 ) i++;
      tnlAssert( i > 0, cerr << " i = " << i  ;)
      const RealType y0 = ySegments[ i - 1 ];
      const RealType y1 = ySegments[ i ];
      const RealType x0 = ySegmentsLeftOffsets[ i - 1 ];
      const RealType x1 = ySegmentsRightOffsets[ i - 1 ];
      const RealType x2 = ySegmentsLeftOffsets[ i ];
      const RealType x3 = ySegmentsRightOffsets[ i ];
      const RealType r = ( y - y0 ) / ( y1 - y0 );
      const RealType x4 = x0 + ( x2 - x0 ) * r;
      const RealType x5 = x1 + ( x3 - x1 ) * r;
      //cout << coordinates << " => " << x << " => " << M_PI * x / proportions. x()<< endl;
      //const RealType xParameter =  M_PI * ( x / proportions. x() - 0.5 );
      vertex. x() = origin. x() + x4 + ( x5 - x4 ) * ( x / proportions. x() );
      vertex. y() = origin. y() + y;
   }
   //vertex. x() = origin. x() + ( coordinates. x() + 0.5 * ( 1 + dx ) ) * parametricStep. x();
   //vertex. x() = origin. x() + ( coordinates. x() + 0.5 * ( 1 + dx ) ) * parametricStep. x();
}

template< typename Real,
          typename Device,
          typename Index >
void tnlLinearGridGeometry< 2, Real, Device, Index > :: setNumberOfSegments( const IndexType segments )
{
   tnlAssert( segments >= 0, cerr << " segments = " << segments );
   ySegments. setSize( segments );
   ySegmentsLeftOffsets. setSize( segments );
   ySegmentsRightOffsets. setSize( segments );
   this -> numberOfSegments = segments;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlLinearGridGeometry< 2, Real, Device, Index > :: setSegmentData( const IndexType segment,
                                                                        const RealType& segmentHeight,
                                                                        const RealType& leftOffset,
                                                                        const RealType& rightOffset )
{
   tnlAssert( segment >= 0 && ( segment < this -> numberOfSegments || this -> numberOfSegments == 0 ),
              cerr << " segment = " << segment << ", this -> numberOfSegments = " << this -> numberOfSegments );
   ySegments[ segment ] = segmentHeight;
   ySegmentsLeftOffsets[ segment ] = leftOffset;
   ySegmentsRightOffsets[ segment ] = rightOffset;
}


template< typename Real,
          typename Device,
          typename Index >
bool tnlLinearGridGeometry< 2, Real, Device, Index > :: save( tnlFile& file ) const
{
   if( ! this -> parametricStep. save( file ) ||
       ! this -> proportions. save( file ) ||
       ! file. write< IndexType, DeviceType >( &this -> numberOfSegments ) ||       
       ! this -> ySegments. save( file ) ||
       ! this -> ySegmentsLeftOffsets. save( file ) ||
       ! this -> ySegmentsRightOffsets. save( file ) )
      return false;
   return true;
};

template< typename Real,
          typename Device,
          typename Index >
bool tnlLinearGridGeometry< 2, Real, Device, Index > :: load( tnlFile& file )
{
   if( ! this -> parametricStep. load( file ) ||
       ! this -> proportions. load( file ) ||
       ! file. read< IndexType, DeviceType >( &this -> numberOfSegments ) ||
       ! this -> ySegments. load( file ) ||
       ! this -> ySegmentsLeftOffsets. load( file ) ||
       ! this -> ySegmentsRightOffsets. load( file ) )
      return false;
   return true;
};




#endif /* TNLLINEARGRIDGEOMETRY_IMPL_H_ */
