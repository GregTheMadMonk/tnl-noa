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

#include <mesh/tnlTraversal.h>
#include <core/vectors/tnlVector.h>
#include <functions/tnlFunctionDiscretizer.h>
#include <matrices/tnlCSRMatrix.h>
#include <matrices/tnlMatrixSetter.h>
#include <solvers/pde/tnlLinearSystemAssembler.h>

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
   Vector functionData, exactData, approximateData;
   const IndexType entities = mesh.getNumberOfCells();
   BoundaryConditionsType boundaryConditions;
   ConstantFunctionType zeroFunction;

   if( ! functionData.setSize( entities ) ||
       ! exactData.setSize( entities ) ||
       ! approximateData.setSize( entities )  )
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
   cout << "Function = " << functionData << endl;
   cout << "Approximation = " << approximateData << endl;
   cout << "Exact = " << exactData << endl;


   l1Err = mesh.getDifferenceLpNorm( exactData, approximateData, ( RealType ) 1.0 );
   l2Err = mesh.getDifferenceLpNorm( exactData, approximateData, ( RealType ) 2.0 );
   maxErr = mesh.getDifferenceAbsMax( exactData, approximateData );
}

/*template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename ExactOperator,
          typename ApproximateOperator,
          typename Function >
void
tnlApproximationError< tnlGrid< Dimensions, Real, Device, Index >, ExactOperator, ApproximateOperator, Function, tnlExplicitApproximation >::
getError( const MeshType& mesh,
          const ExactOperator& exactOperator,
          const ApproximateOperator& approximateOperator,
          const Function& function,
          RealType& l1Err,
          RealType& l2Err,
          RealType& maxErr )
{
   typedef tnlVector< RealType, DeviceType, IndexType > Vector;
   Vector functionData, exactData, approximateData;
   const IndexType entities = mesh.getNumberOfCells();

   if( ! functionData.setSize( entities ) ||
       ! exactData.setSize( entities ) ||
       ! approximateData.setSize( entities )  )
      return;

   tnlFunctionDiscretizer< MeshType, Function, Vector >::template discretize< 0, 0, 0 >( mesh, function, functionData );

   if( DeviceType::DeviceType == ( int ) tnlHostDevice )
   {
      for( IndexType i = 0; i < entities; i++ )
      {
         if( ! mesh.isBoundaryCell( i ) )
         {
            VertexType v = mesh.getCellCenter( i );
            CoordinatesType c = mesh.getCellCoordinates( i );
            exactData[ i ] = exactOperator.getValue( function, v );
            approximateData[ i ] = approximateOperator.getValue( mesh, i, c, functionData, 0.0 );
         }
         else exactData[ i ] = approximateData[ i ];
      }
   }
   if( DeviceType::DeviceType == ( int ) tnlCudaDevice )
   {
      // TODO
   }
   cout << "Function = " << functionData << endl;
   cout << "Approximation = " << approximateData << endl;
   cout << "Exact = " << exactData << endl;

   l1Err = mesh.getDifferenceLpNorm( exactData, approximateData, ( RealType ) 1.0 );
   l2Err = mesh.getDifferenceLpNorm( exactData, approximateData, ( RealType ) 2.0 );
   maxErr = mesh.getDifferenceAbsMax( exactData, approximateData );
}*/

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
   typedef typename MatrixType::RowLengthsVector RowLengthsVectorType;
   Vector functionData, exactData, approximateData;
   MatrixType matrix;
   RowLengthsVectorType rowLengths;
   BoundaryConditionsType boundaryConditions;
   ConstantFunctionType zeroFunction;

   const IndexType entities = mesh.getNumberOfCells();

   cerr << "Semi-implicit test " << endl;

   if( ! functionData.setSize( entities ) ||
       ! exactData.setSize( entities ) ||
       ! approximateData.setSize( entities ) ||
       ! rowLengths.setSize( entities ) )
      return;

   tnlFunctionDiscretizer< Mesh, Function, Vector >::template discretize< 0, 0, 0 >( mesh, function, functionData );

   tnlMatrixSetter< MeshType, ApproximateOperator, BoundaryConditionsType, RowLengthsVectorType > matrixSetter;
   matrixSetter.template getRowLengths< Mesh::Dimensions >( mesh,
                                                            approximateOperator,
                                                            boundaryConditions,
                                                            rowLengths );
   matrix.setDimensions( entities, entities );
   if( ! matrix.setRowLengths( rowLengths ) )
      return;

   tnlLinearSystemAssembler< Mesh, Vector, ApproximateOperator, BoundaryConditionsType, ConstantFunctionType, MatrixType > systemAssembler;
   systemAssembler.template assembly< Mesh::Dimensions >( 0.0, // time
                                                          0.0, // tau
                                                          mesh,
                                                          approximateOperator,
                                                          boundaryConditions,
                                                          zeroFunction,
                                                          functionData,
                                                          matrix,
                                                          approximateData // this has no meaning here
                                                          );

   cout << "Matrix = " << matrix << endl;
   matrix.vectorProduct( functionData, approximateData );
   cout << "Function = " << functionData << endl;
   cout << "Approximation = " << approximateData << endl;
   cout << "Exact = " << exactData << endl;
   for( IndexType i = 0; i < entities; i++ )
      if( mesh.isBoundaryCell( i ) )
         approximateData.setElement( i, exactData.getElement( i ) );

   l1Err = mesh.getDifferenceLpNorm( exactData, approximateData, ( RealType ) 1.0 );
   l2Err = mesh.getDifferenceLpNorm( exactData, approximateData, ( RealType ) 2.0 );
   maxErr = mesh.getDifferenceAbsMax( exactData, approximateData );
}

#endif /* TNLAPPROXIMATIONERROR_IMPL_H_ */
