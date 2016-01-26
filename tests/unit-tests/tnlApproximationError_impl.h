/***************************************************************************
                          tnlApproximationError_impl.h  -  description
                             -------------------
    begin                : Aug 8, 2014
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

#ifndef TNLAPPROXIMATIONERROR_IMPL_H_
#define TNLAPPROXIMATIONERROR_IMPL_H_

#include <mesh/tnlTraverser.h>
#include <core/vectors/tnlVector.h>
#include <matrices/tnlCSRMatrix.h>
#include <matrices/tnlMatrixSetter.h>
#include <solvers/pde/tnlLinearSystemAssembler.h>
#include <solvers/pde/tnlNoTimeDiscretisation.h>
#include <functions/tnlExactOperatorFunction.h>
#include <functions/tnlMeshFunction.h>
#include <functions/tnlOperatorFunction.h>
#include <functions/tnlExactOperatorFunction.h>
#include <solvers/pde/tnlBoundaryConditionsSetter.h>


template< typename ExactOperator,
          typename ApproximateOperator,
          typename MeshEntity,
          typename Function,
          bool writeFunctions >
void
tnlApproximationError< ExactOperator, ApproximateOperator, MeshEntity, Function, writeFunctions >::
getError( const ExactOperator& exactOperator,
          const ApproximateOperator& approximateOperator,
          const Function& function,
          const MeshType& mesh,
          RealType& l1Err,
          RealType& l2Err,
          RealType& maxErr )
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
   if( writeFunctions )
      mesh.save( "mesh-" + meshSizeString + ".tnl" );
   
   //cerr << "Evaluating exact u... " << endl;
   exactU = exactOperatorFunction;
   if( writeFunctions )
      exactU.save( "exact-result-" + meshSizeString + ".tnl" );
   
   //cerr << "Projecting test function ..." << endl;
   v = function;
   if( writeFunctions )
      v.save( "test-function-" + meshSizeString + ".tnl" ) ;
   
   //cerr << "Evaluating approximate u ... " << endl;
   u = operatorFunction;
   tnlBoundaryConditionsSetter< MeshFunction, DirichletBoundaryConditions >::template apply< MeshEntity >( boundaryConditions, 0.0, u );
   if( writeFunctions )
      u.save( "approximate-result-" + meshSizeString + ".tnl" ) ;

   //cerr << "Evaluate difference ... " << endl;
   u -= exactU;   
   tnlBoundaryConditionsSetter< MeshFunction, DirichletBoundaryConditions >::template apply< MeshEntity >( boundaryConditions, 0.0, u );
   if( writeFunctions )
      u.save( "difference-" + meshSizeString + ".tnl" ) ;
   l1Err = u.getLpNorm( 1.0 );
   l2Err = u.getLpNorm( 2.0 );   
   maxErr = u.getMaxNorm();
}

#endif /* TNLAPPROXIMATIONERROR_IMPL_H_ */
