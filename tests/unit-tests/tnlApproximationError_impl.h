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
#include <functions/tnlExactOperatorFunction.h>
#include <functions/tnlMeshFunction.h>
#include <functions/tnlOperatorFunction.h>
#include <functions/tnlExactOperatorFunction.h>
#include <functions/tnlBoundaryOperatorFunction.h>
#include <solvers/pde/tnlBoundaryConditionsSetter.h>


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
   typedef tnlDirichletBoundaryConditions< MeshType, tnlConstantFunction< MeshType::meshDimensions > > DirichletBoundaryConditions;
   typedef tnlBoundaryOperatorFunction< DirichletBoundaryConditions, MeshFunction > BoundaryOperatorFunction;
   typedef tnlOperatorFunction< ApproximateOperator, MeshFunction > OperatorFunction;
   typedef tnlExactOperatorFunction< ExactOperator, Function > ExactOperatorFunction;
   
   tnlMeshFunction< MeshType > exactU( mesh ), u( mesh ), v( mesh );
   OperatorFunction operatorFunction( approximateOperator, v );
   ExactOperatorFunction exactOperatorFunction( exactOperator, function );
   DirichletBoundaryConditions boundaryConditions;
   BoundaryOperatorFunction boundaryOperatorFunction( boundaryConditions, u );

   mesh.save( "mesh.tnl" );
   
   //cerr << "Evaluating exact u... " << endl;
   exactU = exactOperatorFunction;
   exactU.save( "exact-u.tnl" );
   
   //cerr << "Projecting test function ..." << endl;
   v.getData().setValue( 1000.0 );
   v = function;   
   v.save( "function.tnl" ) ;
   
   //cerr << "Evaluating approximate u ... " << endl;
   u = operatorFunction;
   //cerr << " u = " << u.getData() << endl;
   tnlBoundaryConditionsSetter< MeshFunction, DirichletBoundaryConditions >::apply( boundaryConditions, 0.0, u );
   u.save( "approximate-u.tnl" ) ;

   //cerr << "Evaluate difference ... " << endl;
   u -= exactU;   
   //cerr << "Reseting boundary entities ..." << endl;
   tnlBoundaryConditionsSetter< MeshFunction, DirichletBoundaryConditions >::apply( boundaryConditions, 0.0, u );
   //cerr << " u = " << u.getData() << endl;
   u.save( "diff-u.tnl" ) ;
   l1Err = u.getLpNorm( 1.0 );
   l2Err = u.getLpNorm( 2.0 );   
   maxErr = u.getMaxNorm();
   cerr << "l1 = " << l1Err << " l2 = " << l2Err << " maxErr = " << maxErr << endl;
}

#endif /* TNLAPPROXIMATIONERROR_IMPL_H_ */
