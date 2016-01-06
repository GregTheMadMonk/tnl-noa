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
#include <functions/tnlFunctionDiscretizer.h>
#include <matrices/tnlCSRMatrix.h>
#include <matrices/tnlMatrixSetter.h>
#include <solvers/pde/tnlLinearSystemAssembler.h>
#include <solvers/pde/tnlNoTimeDiscretisation.h>
#include <operators/tnlExactOperatorEvaluator.h>
#include <functions/tnlMeshFunction.h>
#include <functions/tnlOperatorFunction.h>
#include <functions/tnlExactOperatorFunction.h>


template< typename ExactOperator,
          typename ApproximateOperator,
          typename Function >
void
tnlApproximationError< ExactOperator, ApproximateOperator, Function >::
getError( const ExactOperator& exactOperator,
          const ApproximateOperator& approximateOperator,
          const Function& function,
          const MeshType& mesh,
          RealType& l1Err,
          RealType& l2Err,
          RealType& maxErr )
{
   typedef tnlMeshFunction< MeshType > MeshFunction;
   typedef tnlDirichletBoundaryConditions< MeshType, tnlConstantFunction< MeshType::meshDimensions > > BoundaryConditions;
   typedef tnlOperatorFunction< ApproximateOperator, Function > OperatorFunction;
   typedef tnlExactOperatorFunction< ExactOperator, Function > ExactOperatorFunction;
   
   tnlMeshFunction< MeshType > exactU( mesh ), u( mesh ), v( mesh );
   BoundaryConditions boundaryConditions;
   OperatorFunction operatorFunction( approximateOperator, function );
   ExactOperatorFunction exactOperatorFunction( exactOperator, function );

   exactU = exactOperatorFunction;
   u = function;

   u.save( "function" ) ;
   
   u = operatorFunction;

   exactU.save( "exact-u" );
   u.save( "approximate-u" ) ;

   u -= exactU;   
   boundaryConditions.apply( u );
   u.save( "diff-u" ) ;
   l1Err = u.getLpNorm( 1.0 );
   l2Err = u.getLpNorm( 2.0 );   
   maxErr = u.getMaxNorm();
}

#endif /* TNLAPPROXIMATIONERROR_IMPL_H_ */
