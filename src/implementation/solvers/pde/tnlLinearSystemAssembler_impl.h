/***************************************************************************
                          tnlLinearSystemAssembler_impl.h  -  description
                             -------------------
    begin                : Oct 12, 2014
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

#ifndef TNLLINEARSYSTEMASSEMBLER_IMPL_H_
#define TNLLINEARSYSTEMASSEMBLER_IMPL_H_

#include <mesh/tnlTraversal_Grid1D.h>
#include <mesh/tnlTraversal_Grid2D.h>
#include <mesh/tnlTraversal_Grid3D.h>

template< typename Mesh,
          typename DofVector,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
   template< int EntityDimensions >
void
tnlLinearSystemAssembler< Mesh, DofVector, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
assembly( const RealType& time,
          const RealType& tau,
          const Mesh& mesh,
          const DifferentialOperator& differentialOperator,
          const BoundaryConditions& boundaryConditions,
          const RightHandSide& rightHandSide,
          DofVector& u,
          MatrixType& matrix,
          DofVector& b ) const
{
   TraversalUserData userData( time, tau, differentialOperator, boundaryConditions, rightHandSide, u, matrix, b );
   TraversalBoundaryEntitiesProcessor boundaryEntitiesProcessor;
   TraversalInteriorEntitiesProcessor interiorEntitiesProcessor;
   tnlTraversal< MeshType, EntityDimensions > meshTraversal;
   meshTraversal.template processEntities< TraversalUserData,
                                           TraversalBoundaryEntitiesProcessor,
                                           TraversalInteriorEntitiesProcessor >
                                          ( mesh,
                                            userData,
                                            boundaryEntitiesProcessor,
                                            interiorEntitiesProcessor );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename DofVector,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
   template< int EntityDimensions >
void
tnlLinearSystemAssembler< tnlGrid< Dimensions, Real, Device, Index >, DofVector, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
assembly( const RealType& time,
          const RealType& tau,
          const tnlGrid< Dimensions, Real, Device, Index >& mesh,
          const DifferentialOperator& differentialOperator,
          const BoundaryConditions& boundaryConditions,
          const RightHandSide& rightHandSide,
          DofVector& u,
          MatrixType& matrix,
          DofVector& b ) const
{
   const IndexType maxRowLength = matrix.getMaxRowLength();
   tnlAssert( maxRowLength > 0, );
   typename TraversalUserData::RowValuesType values;
   typename TraversalUserData::RowColumnsType columns;
   values.setSize( maxRowLength );
   columns.setSize( maxRowLength );

   TraversalUserData userData( time, tau, differentialOperator, boundaryConditions, rightHandSide, u, matrix, b, columns, values );
   TraversalBoundaryEntitiesProcessor boundaryEntitiesProcessor;
   TraversalInteriorEntitiesProcessor interiorEntitiesProcessor;
   tnlTraversal< MeshType, EntityDimensions > meshTraversal;
   meshTraversal.template processEntities< TraversalUserData,
                                           TraversalBoundaryEntitiesProcessor,
                                           TraversalInteriorEntitiesProcessor >
                                          ( mesh,
                                            userData,
                                            boundaryEntitiesProcessor,
                                            interiorEntitiesProcessor );
}


#endif /* TNLLINEARSYSTEMASSEMBLER_IMPL_H_ */
