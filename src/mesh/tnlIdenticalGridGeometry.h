/***************************************************************************
                          tnlIdenticalGridGeometry.h  -  description
                             -------------------
    begin                : Apr 28, 2013
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

#ifndef TNLIDENTICALGRIDGEOMETRY_H_
#define TNLIDENTICALGRIDGEOMETRY_H_

#include <core/tnlHost.h>
#include <core/tnlFeature.h>

template< int Dimensions,
          typename Real = double,
          typename Device = tnlHost,
          typename Index = int >
class tnlIdenticalGridGeometry
{
};

template< typename Real,
          typename Device,
          typename Index >
class tnlIdenticalGridGeometry< 1, Real, Device, Index >
{
   public:

   enum { Dimensions = 1};
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlStaticVector< Dimensions, Index > CoordinatesType;
   typedef tnlStaticVector< Dimensions, Real > VertexType;

   static tnlString getType();

   tnlString getTypeVirtual() const;

   void setParametricStep( const VertexType& parametricStep );

   const VertexType& getParametricStep() const;

   void setProportions( const VertexType& proportions );

   const VertexType& getProportions() const;

   template< typename Vertex >
   void getElementCenter( const VertexType& origin,
                          const CoordinatesType& coordinates,
                          Vertex& center ) const;

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
class tnlIdenticalGridGeometry< 2, Real, Device, Index >
{
   public:

   enum { Dimensions = 2};
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlStaticVector< Dimensions, Index > CoordinatesType;
   typedef tnlStaticVector< Dimensions, Real > VertexType;
   typedef tnlFeature< false > ElementsMeasureStorage;
   typedef tnlFeature< false > DualElementsMeasureStorage;
   typedef tnlFeature< false > EdgeNormalsStorage;
   typedef tnlFeature< false > VerticesStorage;

   static tnlString getType();

   tnlString getTypeVirtual() const;

   void setParametricStep( const VertexType& parametricStep );

   const VertexType& getParametricStep() const;

   void setProportions( const VertexType& proportions );

   const VertexType& getProportions() const;

   template< typename Vertex >
   void getElementCenter( const VertexType& origin,
                          const CoordinatesType& coordinates,
                          Vertex& center ) const;

   Real getElementMeasure( const CoordinatesType& coordinates ) const;

   template< int dx, int dy >
   Real getDualElementMeasure( const CoordinatesType& coordinates ) const;

   template< Index dx, Index dy >
   void getEdgeNormal( const CoordinatesType& coordinates,
                       VertexType& normal ) const;

   template< Index dx, Index dy >
   void getVertex( const CoordinatesType& coordinates,
                   const VertexType& origin,
                   VertexType& vertex ) const;

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
class tnlIdenticalGridGeometry< 3, Real, Device, Index >
{
   public:

   enum { Dimensions = 3};
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlStaticVector< Dimensions, Index > CoordinatesType;
   typedef tnlStaticVector< Dimensions, Real > VertexType;
   typedef tnlFeature< false > ElementsMeasureStorage;
   typedef tnlFeature< false > DualElementsMeasureStorage;
   typedef tnlFeature< false > EdgeNormalsStorage;
   typedef tnlFeature< false > VerticesStorage;

   static tnlString getType();

   tnlString getTypeVirtual() const;

   void setParametricStep( const VertexType& parametricStep );

   const VertexType& getParametricStep() const;

   void setProportions( const VertexType& proportions );

   const VertexType& getProportions() const;

   template< typename Vertex >
   void getElementCenter( const VertexType& origin,
                          const CoordinatesType& coordinates,
                          Vertex& center ) const;

   Real getElementMeasure( const CoordinatesType& coordinates ) const;

   template< int dx, int dy, int dz >
   Real getDualElementMeasure( const CoordinatesType& coordinates ) const;

   template< Index dx, Index dy, Index dz >
   void getEdgeNormal( const CoordinatesType& coordinates,
                       VertexType& normal ) const;

   template< Index dx, Index dy, Index dz >
   void getVertex( const CoordinatesType& coordinates,
                   const VertexType& origin,
                   VertexType& vertex ) const;

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file );

   protected:

   VertexType parametricStep;

   VertexType proportions;

   Real elementMeasure;
};


#include <implementation/mesh/tnlIdenticalGridGeometry_impl.h>

#endif /* TNLIDENTICALGRIDGEOMETRY_H_ */
