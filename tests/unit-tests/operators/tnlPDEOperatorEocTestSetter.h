/***************************************************************************
                          tnlPDEOperatorEocTestSetter.h  -  description
                             -------------------
    begin                : Sep 4, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLPDEOPERATOREOCTESTSETTER_H_
#define TNLPDEOPERATOREOCTESTSETTER_H_

#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/Analytic/ExpBump.h>

using namespace TNL;

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
                                   Meshes::Grid< 1, Real, Device, Index >,
                                   ExpBump< 1, Real > >
{
   public:
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Meshes::Grid< 1, Real, Device, Index > MeshType;
      typedef ExactOperator ExactOperatorType;
      typedef ApproximateOperator ApproximateOperatorType;
      typedef typename MeshType::PointType PointType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef ExpBump< 1, Real > FunctionType;

   static void setMesh( MeshType& mesh,
                        const IndexType& size )
   {
      PointType origin, proportions;
      origin.x() = -2.0;
      proportions.x() = 4.0;
      mesh.setDomain( origin, proportions );

      CoordinatesType dimensions;
      dimensions.x() = size;
      mesh.setDimensions( dimensions );
   };

   static void setFunction( FunctionType& function )
   {
      function.setAmplitude( 1.5 );
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
                                   Meshes::Grid< 2, Real, Device, Index >,
                                   ExpBump< 2, Real > >
{
   public:
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Meshes::Grid< 2, Real, Device, Index > MeshType;
      typedef ExactOperator ExactOperatorType;
      typedef ApproximateOperator ApproximateOperatorType;
      typedef typename MeshType::PointType PointType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef ExpBump< 2, Real > FunctionType;

   static void setMesh( MeshType& mesh,
                        const IndexType& size )
   {
      PointType origin, proportions;
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
                                   Meshes::Grid< 3, Real, Device, Index >,
                                   ExpBump< 3, Real > >
{
   public:
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Meshes::Grid< 3, Real, Device, Index > MeshType;
      typedef ExactOperator ExactOperatorType;
      typedef ApproximateOperator ApproximateOperatorType;
      typedef typename MeshType::PointType PointType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef ExpBump< 3, Real > FunctionType;

   static void setMesh( MeshType& mesh,
                        const IndexType& size )
   {
      PointType origin, proportions;
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
