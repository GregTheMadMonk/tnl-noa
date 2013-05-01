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
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   void setParametricStep( const tnlTuple< 2, Real >& parametricStep );

   void getElementCoordinates( const Index j,
                               const Index i,
                               tnlTuple< 2, Real >& coordinates ) const;

   Real getElementMeasure( const Index j,
                           const Index i ) const;

   template< Index dy, Index dx >
   Real getElementsDistance( const Index j,
                             const Index i ) const;

   template< Index dy, Index dx >
   void getEdgeCoordinates( const Index j,
                            const Index i,
                            tnlTuple< 2, Real >& coordinates ) const

   template< Index dy, Index dx >
   Real getEdgeLength( const Index j,
                       const Index i ) const;

   template< Index dy, Index dx >
   tnlTuple< 2, Real > getEdgeNormal( const Index j,
                                      const Index i ) const;

   void getVertexCoordinates( const Index j,
                              const Index i,
                              tnlTuple< 2, Real >& coordinates ) const;

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file );

   protected:

   tnlTuple< Dimensions, Real > parametricStep;

   Real elementMeasure;

};

#include <implementation/mesh/tnlIdenticalGridGeometry_impl.h>

#endif /* TNLIDENTICALGRIDGEOMETRY_H_ */
