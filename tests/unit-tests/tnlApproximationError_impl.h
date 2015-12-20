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
#include <functors/tnlFunctionDiscretizer.h>
#include <matrices/tnlCSRMatrix.h>
#include <matrices/tnlMatrixSetter.h>
#include <solvers/pde/tnlLinearSystemAssembler.h>
#include <solvers/pde/tnlNoTimeDiscretisation.h>
#include <operators/tnlExactOperatorEvaluator.h>

template< typename Mesh,
          typename ExactOperator,
          typename ApproximateOperator,
          typename Function >
void
tnlApproximationError< Mesh, ExactOperator, ApproximateOperator, Function, tnlExplicitApproximation >::
getError( const Mesh& mesh,
          const ExactOperator& exactOperator,
          const ApproximateOperator& approximateOperator,
          const Function& function,
          RealType& l1Err,
          RealType& l2Err,
          RealType& maxErr )
{
   typedef tnlVector< RealType, DeviceType, IndexType > Vector;
   Vector functionData, exactData, approximateData, aux;
   const IndexType entities = mesh.template getEntitiesCount< typename Mesh::Cell >();
   BoundaryConditionsType boundaryConditions;
   boundaryConditions.setFunction( function );
   ConstantFunctionType zeroFunction;

   if( ! functionData.setSize( entities ) ||
       ! exactData.setSize( entities ) ||
       ! approximateData.setSize( entities ) ||
       ! aux.setSize( entities) )
      return;

   tnlFunctionDiscretizer< Mesh, Function, Vector >::template discretize< 0, 0, 0 >( mesh, function, functionData );

   tnlExplicitUpdater< Mesh, Vector, ApproximateOperator, BoundaryConditionsType, ConstantFunctionType > explicitUpdater;
   explicitUpdater.template update< Mesh::Dimensions >( 0.0,
                                                        mesh,
                                                        approximateOperator,
                                                        boundaryConditions,
                                                        zeroFunction,
                                                        functionData,
                                                        approximateData );
   tnlExactOperatorEvaluator< Mesh, Vector, ExactOperator, Function, BoundaryConditionsType > operatorEvaluator;
   operatorEvaluator.template evaluate< Mesh::Dimensions >( 0.0, mesh, exactOperator, function, boundaryConditions, exactData );

   typename Mesh::template GridEntity< typename Mesh::Cell > cell( mesh );
   for( cell.getCoordinates().x() = 0;
        cell.getCoordinates().x() < entities;
        cell.getCoordinates().x()++ )
   {
      cell.setIndex( mesh.getEntityIndex( cell ) );
      if( cell.isBoundaryEntity() )
         approximateData.setElement( cell.getIndex(), exactData.getElement( cell.getIndex() ) );
   }

   l1Err = mesh.getDifferenceLpNorm( exactData, approximateData, ( RealType ) 1.0 );
   l2Err = mesh.getDifferenceLpNorm( exactData, approximateData, ( RealType ) 2.0 );
   maxErr = mesh.getDifferenceAbsMax( exactData, approximateData );
}

/****
 * Implicit (matrix) approximation
 */

template< typename Mesh,
          typename ExactOperator,
          typename ApproximateOperator,
          typename Function >
void
tnlApproximationError< Mesh, ExactOperator, ApproximateOperator, Function, tnlImplicitApproximation >::
getError( const Mesh& mesh,
          const ExactOperator& exactOperator,
          const ApproximateOperator& approximateOperator,
          const Function& function,
          RealType& l1Err,
          RealType& l2Err,
          RealType& maxErr )
{
   typedef tnlVector< RealType, DeviceType, IndexType > Vector;
   typedef tnlCSRMatrix< RealType, DeviceType, IndexType > MatrixType;
   typedef typename MatrixType::CompressedRowsLengthsVector CompressedRowsLengthsVectorType;
   Vector functionData, exactData, approximateData;
   MatrixType matrix;
   CompressedRowsLengthsVectorType rowLengths;
   BoundaryConditionsType boundaryConditions;
   boundaryConditions.setFunction( function );
   ConstantFunctionType zeroFunction;

   const IndexType entities = mesh.template getEntitiesCount< typename Mesh::Cell >();

   if( ! functionData.setSize( entities ) ||
       ! exactData.setSize( entities ) ||
       ! approximateData.setSize( entities ) ||
       ! rowLengths.setSize( entities ) )
      return;

   tnlFunctionDiscretizer< Mesh, Function, Vector >::template discretize< 0, 0, 0 >( mesh, function, functionData );

   tnlMatrixSetter< MeshType, ApproximateOperator, BoundaryConditionsType, CompressedRowsLengthsVectorType > matrixSetter;
   matrixSetter.template getCompressedRowsLengths< Mesh::Dimensions >( mesh,
                                                            approximateOperator,
                                                            boundaryConditions,
                                                            rowLengths );
   matrix.setDimensions( entities, entities );
   if( ! matrix.setCompressedRowsLengths( rowLengths ) )
      return;

   tnlLinearSystemAssembler< Mesh, Vector, ApproximateOperator, BoundaryConditionsType, ConstantFunctionType, tnlNoTimeDiscretisation, MatrixType > systemAssembler;
   systemAssembler.template assembly< Mesh::Dimensions >( 0.0, // time
                                                          1.0, // tau
                                                          mesh,
                                                          approximateOperator,
                                                          boundaryConditions,
                                                          zeroFunction,
                                                          functionData,
                                                          matrix,
                                                          approximateData // this has no meaning here
                                                          );

   tnlExactOperatorEvaluator< Mesh, Vector, ExactOperator, Function, BoundaryConditionsType > operatorEvaluator;
   operatorEvaluator.template evaluate< Mesh::Dimensions >( 0.0, mesh, exactOperator, function, boundaryConditions, exactData );

   typename Mesh::template GridEntity< typename Mesh::Cell > cell( mesh );
   for( cell.getCoordinates().x() = 0;
        cell.getCoordinates().x() < entities;
        cell.getCoordinates().x()++ )
   {
      IndexType i = mesh.getEntityIndex( cell );
      if( ! cell.isBoundaryEntity() )
         matrix.setElement( i, i, matrix.getElement( i, i ) - 1.0 );
   }
   matrix.vectorProduct( functionData, approximateData );

   // TODO: replace this when matrix.vectorProduct has multiplicator parameter
   for( IndexType i = 0; i < entities; i++ )
      approximateData.setElement( i, -1.0 * approximateData.getElement( i ) );

   for( cell.getCoordinates().x() = 0;
        cell.getCoordinates().x() < entities;
        cell.getCoordinates().x()++ )
   {
      IndexType i = mesh.getEntityIndex( cell );
      if( cell.isBoundaryEntity() )
         approximateData.setElement( i, exactData.getElement( i ) );
   }

   l1Err = mesh.getDifferenceLpNorm( exactData, approximateData, ( RealType ) 1.0 );
   l2Err = mesh.getDifferenceLpNorm( exactData, approximateData, ( RealType ) 2.0 );
   maxErr = mesh.getDifferenceAbsMax( exactData, approximateData );
}

#endif /* TNLAPPROXIMATIONERROR_IMPL_H_ */
