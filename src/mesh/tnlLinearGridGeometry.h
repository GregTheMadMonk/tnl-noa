/***************************************************************************
                          tnlLinearGridGeometry.h  -  description
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

#ifndef TNLLINEARGRIDGEOMETRY_H_
#define TNLLINEARGRIDGEOMETRY_H_


#include <core/tnlHost.h>

template< int Dimensions,
          typename Real = double,
          typename Device = tnlHost,
          typename Index = int >
class tnlLinearGridGeometry
{
};

template< typename Real,
          typename Device,
          typename Index >
class tnlLinearGridGeometry< 1, Real, Device, Index >
{
   public:

   enum { Dimensions = 1};
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlTuple< 1, Index > CoordinatesType;
   typedef tnlTuple< 1, Real > VertexType;

   static tnlString getType();

   void setParametricStep( const VertexType& parametricStep );

   const VertexType& getParametricStep() const;

   void setProportions( const VertexType& proportions );

   const VertexType& getProportions() const;

   void getElementCenter( const VertexType& origin,
                          const CoordinatesType& coordinates,
                          VertexType& center ) const;

   Real getElementMeasure( const CoordinatesType& i ) const;

   template< Index dx >
   Real getElementsDistance( const Index i ) const;

   template< Index dx >
   void getEdgeCoordinates( const Index i,
                            const VertexType& origin,
                            VertexType& coordinates ) const;

   template< Index dx >
   Real getEdgeLength( const Index i ) const;

   template< Index dx >
   void getEdgeNormal( const Index i,
                       VertexType& normal ) const;

   void getVertexCoordinates( const Index i,
                              const VertexType& origin,
                              VertexType& coordinates ) const;

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file );

   protected:

   VertexType parametricStep;

   VertexType proportions;

   Real elementMeasure;
};

template< typename Real,
          typename Device,
          typename Index >
class tnlLinearGridGeometry< 2, Real, Device, Index >
{
   public:

   enum { Dimensions = 2};
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlTuple< Dimensions, Index > CoordinatesType;
   typedef tnlTuple< Dimensions, Real > VertexType;
   typedef tnlFeature< true > ElementsMeasureStorage;
   typedef tnlFeature< true > DualElementsMeasureStorage;
   typedef tnlFeature< true > EdgeNormalsStorage;
   typedef tnlFeature< true > VerticesStorage;

   tnlLinearGridGeometry();

   static tnlString getType();

   void setParametricStep( const VertexType& parametricStep );

   const VertexType& getParametricStep() const;

   void setProportions( const VertexType& proportions );

   const VertexType& getProportions() const;

   void getElementCenter( const VertexType& origin,
                          const tnlTuple< Dimensions, Index >& coordinates,
                          VertexType& center ) const;

   Real getElementMeasure( const tnlTuple< Dimensions, Index >& coordinates ) const;

   template< int dx, int dy >
   Real getDualElementMeasure( const CoordinatesType& coordinates ) const;

   template< Index dx, Index dy >
   void getEdgeNormal( const CoordinatesType& coordinates,
                       VertexType& normal ) const;

   template< Index dx, Index dy >
   void getVertex( const CoordinatesType& coordinates,
                   const VertexType& origin,
                   VertexType& vertex ) const;

   void setNumberOfSegments( const IndexType segments );

   void setSegmentData( const IndexType segment,
                        const RealType& segmentHeight,
                        const RealType& leftOffset,
                        const RealType& rightOffset );

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file );

   protected:

   VertexType parametricStep;

   VertexType proportions;

   IndexType numberOfSegments;

   tnlVector< RealType, DeviceType, IndexType > ySegments, ySegmentsLeftOffsets, ySegmentsRightOffsets;
};

template< typename Real,
          typename Device,
          typename Index >
class tnlLinearGridGeometry< 3, Real, Device, Index >
{
   public:

   enum { Dimensions = 3};
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlTuple< Dimensions, Index > CoordinatesType;
   typedef tnlTuple< Dimensions, Real > VertexType;
   typedef tnlFeature< true > ElementsMeasureStorage;
   typedef tnlFeature< true > DualElementsMeasureStorage;
   typedef tnlFeature< true > EdgeNormalsStorage;
   typedef tnlFeature< true > VerticesStorage;

   tnlLinearGridGeometry();

   static tnlString getType();

   void setParametricStep( const VertexType& parametricStep );

   const VertexType& getParametricStep() const;

   void setProportions( const VertexType& proportions );

   const VertexType& getProportions() const;

   void getElementCenter( const VertexType& origin,
                          const tnlTuple< Dimensions, Index >& coordinates,
                          VertexType& center ) const;

   Real getElementMeasure( const tnlTuple< Dimensions, Index >& coordinates ) const;

   template< int dx, int dy >
   Real getDualElementMeasure( const CoordinatesType& coordinates ) const;

   template< Index dx, Index dy >
   void getEdgeNormal( const CoordinatesType& coordinates,
                       VertexType& normal ) const;

   template< Index dx, Index dy >
   void getVertex( const CoordinatesType& coordinates,
                   const VertexType& origin,
                   VertexType& vertex ) const;

   void setNumberOfSegments( const IndexType segments );

   void setSegmentData( const IndexType segment,
                        const RealType& segmentHeight,
                        const RealType& leftOffset,
                        const RealType& rightOffset );

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file );

   protected:

   VertexType parametricStep;

   VertexType proportions;

   IndexType numberOfSegments;

   tnlVector< RealType, DeviceType, IndexType > ySegments, ySegmentsLeftOffsets, ySegmentsRightOffsets;
};


#include <implementation/mesh/tnlLinearGridGeometry_impl.h>


#endif /* TNLLINEARGRIDGEOMETRY_H_ */
