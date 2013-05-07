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

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlTuple< 1, Index > CoordinatesType;
   typedef tnlTuple< 1, Real > VertexType;

   void setParametricStep( const VertexType& parametricStep );

   const VertexType& getParametricStep() const;

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

   Real elementMeasure;
};

template< typename Real,
          typename Device,
          typename Index >
class tnlIdenticalGridGeometry< 2, Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlTuple< 2, Index > CoordinatesType;
   typedef tnlTuple< 2, Real > VertexType;

   void setParametricStep( const VertexType& parametricStep );

   const VertexType& getParametricStep() const;

   void getElementCenter( const VertexType& origin,
                          const tnlTuple< 2, Index >& coordinates,
                          VertexType& center ) const;

   Real getElementMeasure( const tnlTuple< 2, Index >& coordinates ) const;

   Real getElementsDistance( const CoordinatesType& c1,
                             const CoordinatesType& c2 ) const;

   /*template< Index dx, Index dy >
   void getEdgeCoordinates( const Index i,
                            const Index j,
                            const VertexType& origin,
                            VertexType& coordinates ) const;

   template< Index dx, Index dy >
   Real getEdgeLength( const Index i,
                       const Index j ) const;*/

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

   Real elementMeasure;
};

#include <implementation/mesh/tnlIdenticalGridGeometry_impl.h>

#endif /* TNLIDENTICALGRIDGEOMETRY_H_ */
