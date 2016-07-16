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

#include <mesh/tnlGrid.h>
#include <functions/tnlConstantFunction.h>
#include <operators/tnlDirichletBoundaryConditions.h>
#include <solvers/pde/tnlExplicitUpdater.h>
#include <functions/tnlExactOperatorFunction.h>
#include <functions/tnlMeshFunction.h>
#include <solvers/pde/tnlBoundaryConditionsSetter.h>

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
      typedef tnlConstantFunction< MeshType::meshDimensions, RealType > ConstantFunctionType;
      typedef tnlDirichletBoundaryConditions< MeshType, Function  > BoundaryConditionsType;

      static void getError( const ExactOperator& exactOperator,
                            ApproximateOperator& approximateOperator,
                            const Function& function,
                            const MeshType& mesh,
                            RealType& l1Error,
                            RealType& l2Error,
                            RealType& maxError,
                            bool writeFunctions )
      {
         typedef tnlMeshFunction< MeshType, MeshEntity::getDimensions() > MeshFunction;
         typedef tnlDirichletBoundaryConditions< MeshType, tnlConstantFunction< MeshType::meshDimensions > > DirichletBoundaryConditions;
         typedef tnlOperatorFunction< DirichletBoundaryConditions, MeshFunction > BoundaryOperatorFunction;
         typedef tnlOperatorFunction< ApproximateOperator, MeshFunction > OperatorFunction;
         typedef tnlExactOperatorFunction< ExactOperator, Function > ExactOperatorFunction;

         tnlMeshFunction< MeshType, MeshEntity::getDimensions() > exactU( mesh ), u( mesh ), v( mesh );
         OperatorFunction operatorFunction( approximateOperator, v );
         ExactOperatorFunction exactOperatorFunction( exactOperator, function );
         DirichletBoundaryConditions boundaryConditions;
         BoundaryOperatorFunction boundaryOperatorFunction( boundaryConditions, u );

         tnlString meshSizeString( mesh.getDimensions().x() );
         tnlString dimensionsString;
         if( MeshType::getMeshDimensions() == 1 )
            dimensionsString = "1D-";
         if( MeshType::getMeshDimensions() == 2 )
            dimensionsString = "2D-";
         if( MeshType::getMeshDimensions() == 3 )
            dimensionsString = "3D-";

         //if( writeFunctions )
         //   mesh.save( "mesh-" + dimensionsString + meshSizeString + ".tnl" );

         //cerr << "Evaluating exact u... " << endl;
         exactU = exactOperatorFunction;
         if( writeFunctions )
            exactU.write( "exact-result-" + dimensionsString + meshSizeString, "gnuplot" );

         //cerr << "Projecting test function ..." << endl;
         v = function;
         if( writeFunctions )
            v.write( "test-function-" + dimensionsString + meshSizeString, "gnuplot" ) ;

         //cerr << "Evaluating approximate u ... " << endl;
         operatorFunction.setPreimageFunction( v );
         if( ! operatorFunction.deepRefresh() )
         {
            cerr << "Error in operator refreshing." << endl;
            return;
         }
         u = operatorFunction;
         tnlBoundaryConditionsSetter< MeshFunction, DirichletBoundaryConditions >::template apply< MeshEntity >( boundaryConditions, 0.0, u );
         if( writeFunctions )
            u.write( "approximate-result-" + dimensionsString + meshSizeString, "gnuplot" ) ;

         //cerr << "Evaluate difference ... " << endl;
         u -= exactU;
         tnlBoundaryConditionsSetter< MeshFunction, DirichletBoundaryConditions >::template apply< MeshEntity >( boundaryConditions, 0.0, u );
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
      typedef tnlConstantFunction< MeshType::meshDimensions, RealType > ConstantFunctionType;
      typedef tnlDirichletBoundaryConditions< MeshType, Function  > BoundaryConditionsType;

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
