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

#include <type_traits>
#include <mesh/grids/tnlTraverser_Grid1D.h>
#include <mesh/grids/tnlTraverser_Grid2D.h>
#include <mesh/grids/tnlTraverser_Grid3D.h>

template< typename Mesh,
          typename MeshFunction,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename TimeDiscretisation,
          typename Matrix,
          typename DofVector >
   template< typename EntityType >
void
tnlLinearSystemAssembler< Mesh, MeshFunction, DifferentialOperator, BoundaryConditions, RightHandSide, TimeDiscretisation, Matrix, DofVector >::
assembly( const RealType& time,
          const RealType& tau,
          const Mesh& mesh,
          const DifferentialOperator& differentialOperator,
          const BoundaryConditions& boundaryConditions,
          const RightHandSide& rightHandSide,
          const MeshFunction& u,
          MatrixType& matrix,
          DofVector& b ) const
{
      static_assert( std::is_same< MeshFunction, 
                                tnlVector< typename MeshFunction::RealType,
                                           typename MeshFunction::DeviceType,
                                           typename MeshFunction::IndexType > >::value != true,
      "Error: I am getting tnlVector instead of tnlMeshFunction or similar object. You might forget to bind DofVector into tnlMeshFunction in you method getExplicitRHS."  );

   const IndexType maxRowLength = matrix.getMaxRowLength();
   tnlAssert( maxRowLength > 0, );

   if( std::is_same< DeviceType, tnlHost >::value )
   {
      TraverserUserData userData( time,
                                  tau,
                                  differentialOperator,
                                  boundaryConditions,
                                  rightHandSide,
                                  u,
                                  matrix,
                                  b );
      tnlTraverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserBoundaryEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
      meshTraverser.template processInteriorEntities< TraverserUserData,
                                                      TraverserInteriorEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
   }
   if( std::is_same< DeviceType, tnlCuda >::value )
   {
      RealType* kernelTime = tnlCuda::passToDevice( time );
      RealType* kernelTau = tnlCuda::passToDevice( tau );
      DifferentialOperator* kernelDifferentialOperator = tnlCuda::passToDevice( differentialOperator );
      BoundaryConditions* kernelBoundaryConditions = tnlCuda::passToDevice( boundaryConditions );
      RightHandSide* kernelRightHandSide = tnlCuda::passToDevice( rightHandSide );
      MeshFunction* kernelU = tnlCuda::passToDevice( u );
      DofVector* kernelB = tnlCuda::passToDevice( b );
      MatrixType* kernelMatrix = tnlCuda::passToDevice( matrix );
      TraverserUserData userData( *kernelTime,
                                  *kernelTau,
                                  *kernelDifferentialOperator,
                                  *kernelBoundaryConditions,
                                  *kernelRightHandSide,
                                  *kernelU,
                                  *kernelMatrix,
                                  *kernelB );
      checkCudaDevice;
      tnlTraverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserBoundaryEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
      meshTraverser.template processInteriorEntities< TraverserUserData,
                                                      TraverserInteriorEntitiesProcessor >
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
   
   if( std::is_same< DeviceType, tnlMIC >::value )
   {
      const RealType* kernelTime = tnlMIC::passToDevice( time );
      const RealType* kernelTau = tnlMIC::passToDevice( tau );
      const DifferentialOperator* kernelDifferentialOperator = tnlMIC::passToDevice( differentialOperator );
      const BoundaryConditions* kernelBoundaryConditions = tnlMIC::passToDevice( boundaryConditions );
      const RightHandSide* kernelRightHandSide = tnlMIC::passToDevice( rightHandSide );
      const MeshFunction* kernelU = tnlMIC::passToDevice( u );
      DofVector* kernelB = tnlMIC::passToDevice( b );
      MatrixType* kernelMatrix = tnlMIC::passToDevice( matrix );
      TraverserUserData userData( *kernelTime,
                                  *kernelTau,
                                  *kernelDifferentialOperator,
                                  *kernelBoundaryConditions,
                                  *kernelRightHandSide,
                                  *kernelU,
                                  *kernelMatrix,
                                  *kernelB );
      
      tnlTraverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserBoundaryEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
      meshTraverser.template processInteriorEntities< TraverserUserData,
                                                      TraverserInteriorEntitiesProcessor >
                                                    ( mesh,
                                                      userData );

      
      tnlMIC::freeFromDevice( kernelTime );
      tnlMIC::freeFromDevice( kernelTau );
      tnlMIC::freeFromDevice( kernelDifferentialOperator );
      tnlMIC::freeFromDevice( kernelBoundaryConditions );
      tnlMIC::freeFromDevice( kernelRightHandSide );
      tnlMIC::freeFromDevice( kernelU );
      tnlMIC::freeFromDevice( kernelB );
      tnlMIC::freeFromDevice( kernelMatrix );
   }
   
}

#endif /* TNLLINEARSYSTEMASSEMBLER_IMPL_H_ */
