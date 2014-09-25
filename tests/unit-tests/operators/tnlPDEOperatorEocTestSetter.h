/***************************************************************************
                          tnlPDEOperatorEocTestSetter.h  -  description
                             -------------------
    begin                : Sep 4, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLPDEOPERATOREOCTESTSETTER_H_
#define TNLPDEOPERATOREOCTESTSETTER_H_

#include <mesh/tnlGrid.h>
#include <functions/tnlExpBumpFunction.h>

template< typename ApproximateOperator,
          typename ExactOperator,
          typename Mesh,
          typename TestFunction >
class tnlPDEOperatorEocTestSetter
{
};

template< typename ApproximateOperator,
          typename ExactOperator,
          typename Real,
          typename Device,
          typename Index >
class tnlPDEOperatorEocTestSetter< ApproximateOperator,
                                   ExactOperator,
                                   tnlGrid< 1, Real, Device, Index >,
                                   tnlExpBumpFunction< 1, Real > >
{
   public:
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef tnlGrid< 1, Real, Device, Index > MeshType;
      typedef ExactOperator ExactOperatorType;
      typedef ApproximateOperator ApproximateOperatorType;
      typedef typename MeshType::VertexType VertexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef tnlExpBumpFunction< 1, Real > FunctionType;

   static void setMesh( MeshType& mesh,
                        const IndexType& size )
   {
      VertexType origin, proportions;
      origin.x() = -1.0;
      proportions.x() = 2.0;
      mesh.setDomain( origin, proportions );

      CoordinatesType dimensions;
      dimensions.x() = size;
      mesh.setDimensions( dimensions );
   };

   static void setFunction( FunctionType& function )
   {
      function.setAmplitude( 1.0 );
      function.setSigma( 0.5 );
   };
};

template< typename ApproximateOperator,
          typename ExactOperator,
          typename Real,
          typename Device,
          typename Index >
class tnlPDEOperatorEocTestSetter< ApproximateOperator,
                                   ExactOperator,
                                   tnlGrid< 2, Real, Device, Index >,
                                   tnlExpBumpFunction< 2, Real > >
{
   public:
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef tnlGrid< 2, Real, Device, Index > MeshType;
      typedef ExactOperator ExactOperatorType;
      typedef ApproximateOperator ApproximateOperatorType;
      typedef typename MeshType::VertexType VertexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef tnlExpBumpFunction< 2, Real > FunctionType;

   static void setMesh( MeshType& mesh,
                        const IndexType& size )
   {
      VertexType origin, proportions;
      origin.x() = -1.0;
      origin.y() = -1.0;
      proportions.x() = 2.0;
      proportions.y() = 2.0;
      mesh.setDomain( origin, proportions );

      CoordinatesType dimensions;
      dimensions.x() = size;
      dimensions.y() = size;
      mesh.setDimensions( dimensions );
   };

   static void setFunction( FunctionType& function )
   {
      function.setAmplitude( 1.0 );
      function.setSigma( 0.5 );
   };
};

template< typename ApproximateOperator,
          typename ExactOperator,
          typename Real,
          typename Device,
          typename Index >
class tnlPDEOperatorEocTestSetter< ApproximateOperator,
                                   ExactOperator,
                                   tnlGrid< 3, Real, Device, Index >,
                                   tnlExpBumpFunction< 3, Real > >
{
   public:
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef tnlGrid< 3, Real, Device, Index > MeshType;
      typedef ExactOperator ExactOperatorType;
      typedef ApproximateOperator ApproximateOperatorType;
      typedef typename MeshType::VertexType VertexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef tnlExpBumpFunction< 3, Real > FunctionType;

   static void setMesh( MeshType& mesh,
                        const IndexType& size )
   {
      VertexType origin, proportions;
      origin.x() = -1.0;
      origin.y() = -1.0;
      origin.z() = -1.0;
      proportions.x() = 2.0;
      proportions.y() = 2.0;
      proportions.z() = 2.0;
      mesh.setDomain( origin, proportions );

      CoordinatesType dimensions;
      dimensions.x() = size;
      dimensions.y() = size;
      dimensions.z() = size;
      mesh.setDimensions( dimensions );
   };

   static void setFunction( FunctionType& function )
   {
      function.setAmplitude( 1.0 );
      function.setSigma( 0.5 );
   };
};
#endif /* TNLPDEOPERATOREOCTESTSETTER_H_ */
