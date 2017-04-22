/***************************************************************************
                          tnlApproximationError.h  -  description
                             -------------------
    begin                : Aug 7, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLAPPROXIMATIONERROR_H_
#define TNLAPPROXIMATIONERROR_H_

#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/Analytic/Constant.h>
#include <TNL/Operators/DirichletBoundaryConditions.h>
#include <TNL/Solvers/PDE/ExplicitUpdater.h>
#include <TNL/Functions/ExactOperatorFunction.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Solvers/PDE/BoundaryConditionsSetter.h>

using namespace TNL;

template< typename ExactOperator,
          typename ApproximateOperator,
          typename MeshEntity,
          typename Function >
class tnlApproximationError
{
   public:
 
      typedef typename ApproximateOperator::RealType RealType;
      typedef typename ApproximateOperator::MeshType MeshType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      typedef typename MeshType::VertexType VertexType;
      typedef SharedPointer< MeshType > MeshPointer;
      typedef Functions::Analytic::Constant< MeshType::meshDimension, RealType > ConstantType;
      typedef Operators::DirichletBoundaryConditions< MeshType, Function  > BoundaryConditionsType;

      static void getError( const ExactOperator& exactOperator,
                            ApproximateOperator& approximateOperator,
                            const Function& function,
                            const MeshPointer& meshPointer,
                            RealType& l1Error,
                            RealType& l2Error,
                            RealType& maxError,
                            bool writeFunctions )
      {
         typedef Functions::MeshFunction< MeshType, MeshEntity::getDimensions() > MeshFunction;
         typedef Operators::DirichletBoundaryConditions< MeshType, Functions::Analytic::Constant< MeshType::meshDimension > > DirichletBoundaryConditions;
         typedef Functions::OperatorFunction< DirichletBoundaryConditions, MeshFunction > BoundaryOperatorFunction;
         typedef Functions::OperatorFunction< ApproximateOperator, MeshFunction > OperatorFunction;
         typedef Functions::ExactOperatorFunction< ExactOperator, Function > ExactOperatorFunction;

         Functions::MeshFunction< MeshType, MeshEntity::getDimensions() > exactU( meshPointer ), u( meshPointer ), v( meshPointer );
         OperatorFunction operatorFunction( approximateOperator, v );
         ExactOperatorFunction exactOperatorFunction( exactOperator, function );
         DirichletBoundaryConditions boundaryConditions;
         BoundaryOperatorFunction boundaryOperatorFunction( boundaryConditions, u );

         String meshSizeString( meshPointer->getDimensions().x() );
         String dimensionsString;
         if( MeshType::getDimension() == 1 )
            dimensionsString = "1D-";
         if( MeshType::getDimension() == 2 )
            dimensionsString = "2D-";
         if( MeshType::getDimension() == 3 )
            dimensionsString = "3D-";

         //if( writeFunctions )
         //   mesh.save( "mesh-" + dimensionsString + meshSizeString + ".tnl" );

         //cerr << "Evaluating exact u... " << std::endl;
         exactU = exactOperatorFunction;
         if( writeFunctions )
            exactU.write( "exact-result-" + dimensionsString + meshSizeString, "gnuplot" );

         //cerr << "Projecting test function ..." << std::endl;
         v = function;
         if( writeFunctions )
            v.write( "test-function-" + dimensionsString + meshSizeString, "gnuplot" ) ;

         //cerr << "Evaluating approximate u ... " << std::endl;
         operatorFunction.setPreimageFunction( v );
         if( ! operatorFunction.deepRefresh() )
         {
            std::cerr << "Error in operator refreshing." << std::endl;
            return;
         }
         u = operatorFunction;
         Solvers::PDE::BoundaryConditionsSetter< MeshFunction, DirichletBoundaryConditions >::template apply< MeshEntity >( boundaryConditions, 0.0, u );
         if( writeFunctions )
            u.write( "approximate-result-" + dimensionsString + meshSizeString, "gnuplot" ) ;

         //cerr << "Evaluate difference ... " << std::endl;
         u -= exactU;
         Solvers::PDE::BoundaryConditionsSetter< MeshFunction, DirichletBoundaryConditions >::template apply< MeshEntity >( boundaryConditions, 0.0, u );
         if( writeFunctions )
            u.write( "difference-" + dimensionsString + meshSizeString, "gnuplot" ) ;
         l1Error = u.getLpNorm( 1.0 );
         l2Error = u.getLpNorm( 2.0 );
         maxError = u.getMaxNorm();
      }
};

/*
template< typename Mesh,
          typename ExactOperator,
          typename ApproximateOperator,
          typename Function >
class tnlApproximationError< Mesh, ExactOperator, ApproximateOperator, Function, tnlImplicitApproximation >
{
     public:

      typedef typename ApproximateOperator::RealType RealType;
      typedef Mesh MeshType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      typedef typename MeshType::VertexType VertexType;
      typedef Constant< MeshType::meshDimension, RealType > ConstantType;
      typedef DirichletBoundaryConditions< MeshType, Function  > BoundaryConditionsType;

      static void getError( const Mesh& mesh,
                            const ExactOperator& exactOperator,
                            const ApproximateOperator& approximateOperator,
                            const Function& function,
                            RealType& l1Err,
                            RealType& l2Err,
                            RealType& maxErr );
};
*/
//#include "tnlApproximationError_impl.h"

#endif /* TNLAPPROXIMATIONERROR_H_ */
