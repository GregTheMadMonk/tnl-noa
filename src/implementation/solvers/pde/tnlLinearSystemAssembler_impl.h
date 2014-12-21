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
   const IndexType maxRowLength = matrix.getMaxRowLength();
   tnlAssert( maxRowLength > 0, );
   typedef typename TraversalUserData::RowValuesType RowValuesType;
   typedef typename TraversalUserData::RowColumnsType RowColumnsType;
   RowValuesType values;
   RowColumnsType columns;

   if( DeviceType::DeviceType == tnlHostDevice )
   {
      TraversalUserData userData( time, tau, differentialOperator, boundaryConditions, rightHandSide, u, matrix, b );
      tnlTraversal< MeshType, EntityDimensions > meshTraversal;
      meshTraversal.template processBoundaryEntities< TraversalUserData,
                                                      TraversalBoundaryEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
      meshTraversal.template processInteriorEntities< TraversalUserData,
                                                      TraversalInteriorEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
   }
   if( DeviceType::DeviceType == tnlCudaDevice )
   {
      RealType* kernelTime = tnlCuda::passToDevice( time );
      RealType* kernelTau = tnlCuda::passToDevice( tau );
      DifferentialOperator* kernelDifferentialOperator = tnlCuda::passToDevice( differentialOperator );
      BoundaryConditions* kernelBoundaryConditions = tnlCuda::passToDevice( boundaryConditions );
      RightHandSide* kernelRightHandSide = tnlCuda::passToDevice( rightHandSide );
      DofVector* kernelU = tnlCuda::passToDevice( u );
      DofVector* kernelB = tnlCuda::passToDevice( b );
      MatrixType* kernelMatrix = tnlCuda::passToDevice( matrix );
      TraversalUserData userData( *kernelTime, *kernelTau, *kernelDifferentialOperator, *kernelBoundaryConditions, *kernelRightHandSide, *kernelU, *kernelMatrix, *kernelB );
      checkCudaDevice;
      tnlTraversal< MeshType, EntityDimensions > meshTraversal;
      meshTraversal.template processBoundaryEntities< TraversalUserData,
                                                      TraversalBoundaryEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
      meshTraversal.template processInteriorEntities< TraversalUserData,
                                                      TraversalInteriorEntitiesProcessor >
                                                    ( mesh,
                                                      userData );

      checkCudaDevice;
      tnlCuda::freeFromDevice( kernelTime );
      tnlCuda::freeFromDevice( kernelTau );
      tnlCuda::freeFromDevice( kernelDifferentialOperator );
      tnlCuda::freeFromDevice( kernelBoundaryConditions );
      tnlCuda::freeFromDevice( kernelRightHandSide );
      tnlCuda::freeFromDevice( kernelU );
      tnlCuda::freeFromDevice( kernelB );
      tnlCuda::freeFromDevice( kernelMatrix );
      checkCudaDevice;
   }
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

   if( DeviceType::DeviceType == tnlHostDevice )
   {
      TraversalUserData userData( time, tau, differentialOperator, boundaryConditions, rightHandSide, u, matrix, b );
      tnlTraversal< MeshType, EntityDimensions > meshTraversal;
      meshTraversal.template processBoundaryEntities< TraversalUserData,
                                                      TraversalBoundaryEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
      meshTraversal.template processInteriorEntities< TraversalUserData,
                                                      TraversalInteriorEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
   }
   if( DeviceType::DeviceType == tnlCudaDevice )
   {
      RealType* kernelTime = tnlCuda::passToDevice( time );
      RealType* kernelTau = tnlCuda::passToDevice( tau );
      DifferentialOperator* kernelDifferentialOperator = tnlCuda::passToDevice( differentialOperator );
      BoundaryConditions* kernelBoundaryConditions = tnlCuda::passToDevice( boundaryConditions );
      RightHandSide* kernelRightHandSide = tnlCuda::passToDevice( rightHandSide );
      DofVector* kernelU = tnlCuda::passToDevice( u );
      DofVector* kernelB = tnlCuda::passToDevice( b );
      MatrixType* kernelMatrix = tnlCuda::passToDevice( matrix );
      TraversalUserData userData( *kernelTime, *kernelTau, *kernelDifferentialOperator, *kernelBoundaryConditions, *kernelRightHandSide, *kernelU, *kernelMatrix, *kernelB );
      checkCudaDevice;
      tnlTraversal< MeshType, EntityDimensions > meshTraversal;
      meshTraversal.template processBoundaryEntities< TraversalUserData,
                                                      TraversalBoundaryEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
      meshTraversal.template processInteriorEntities< TraversalUserData,
                                                      TraversalInteriorEntitiesProcessor >
                                                    ( mesh,
                                                      userData );

      checkCudaDevice;
      tnlCuda::freeFromDevice( kernelTime );
      tnlCuda::freeFromDevice( kernelTau );
      tnlCuda::freeFromDevice( kernelDifferentialOperator );
      tnlCuda::freeFromDevice( kernelBoundaryConditions );
      tnlCuda::freeFromDevice( kernelRightHandSide );
      tnlCuda::freeFromDevice( kernelU );
      tnlCuda::freeFromDevice( kernelB );
      tnlCuda::freeFromDevice( kernelMatrix );
      checkCudaDevice;
   }
}


#endif /* TNLLINEARSYSTEMASSEMBLER_IMPL_H_ */