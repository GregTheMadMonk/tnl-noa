/***************************************************************************
                          tnlLinearSystemAssembler_impl.h  -  description
                             -------------------
    begin                : Oct 12, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>
#include <TNL/mesh/grids/tnlTraverser_Grid1D.h>
#include <TNL/mesh/grids/tnlTraverser_Grid2D.h>
#include <TNL/mesh/grids/tnlTraverser_Grid3D.h>

namespace TNL {

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
          const MeshPointer& meshPointer,
          const DifferentialOperatorPointer& differentialOperatorPointer,
          const BoundaryConditionsPointer& boundaryConditionsPointer,
          const RightHandSidePointer& rightHandSidePointer,
          const MeshFunctionPointer& uPointer,
          MatrixPointer& matrixPointer,
          DofVectorPointer& bPointer ) const
{
      static_assert( std::is_same< MeshFunction,
                                Vectors::Vector< typename MeshFunction::RealType,
                                           typename MeshFunction::DeviceType,
                                           typename MeshFunction::IndexType > >::value != true,
      "Error: I am getting Vector instead of MeshFunction or similar object. You might forget to bind DofVector into MeshFunction in you method getExplicitRHS."  );

   const IndexType maxRowLength = matrixPointer.template getData< Devices::Host >().getMaxRowLength();
   Assert( maxRowLength > 0, );

   //if( std::is_same< DeviceType, Devices::Host >::value )
   {
      TraverserUserData userData( time,
                                  tau,
                                  differentialOperatorPointer,
                                  boundaryConditionsPointer,
                                  rightHandSidePointer,
                                  uPointer,
                                  matrixPointer,
                                  bPointer );
      tnlTraverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserBoundaryEntitiesProcessor >
                                                    ( meshPointer,
                                                      userData );
      meshTraverser.template processInteriorEntities< TraverserUserData,
                                                      TraverserInteriorEntitiesProcessor >
                                                    ( meshPointer,
                                                      userData );
   }
   /*if( std::is_same< DeviceType, tnlCuda >::value )
   {
      RealType* kernelTime = Devices::Cuda::passToDevice( time );
      RealType* kernelTau = Devices::Cuda::passToDevice( tau );
      DifferentialOperator* kernelDifferentialOperator = Devices::Cuda::passToDevice( differentialOperator );
      BoundaryConditions* kernelBoundaryConditions = Devices::Cuda::passToDevice( boundaryConditions );
      RightHandSide* kernelRightHandSide = Devices::Cuda::passToDevice( rightHandSide );
      MeshFunction* kernelU = Devices::Cuda::passToDevice( u );
      DofVector* kernelB = Devices::Cuda::passToDevice( b );
      MatrixType* kernelMatrix = Devices::Cuda::passToDevice( matrix );
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
                                                    ( meshPointer,
                                                      userData );
      meshTraverser.template processInteriorEntities< TraverserUserData,
                                                      TraverserInteriorEntitiesProcessor >
                                                    ( meshPointer,
                                                      userData );

      checkCudaDevice;
      Devices::Cuda::freeFromDevice( kernelTime );
      Devices::Cuda::freeFromDevice( kernelTau );
      Devices::Cuda::freeFromDevice( kernelDifferentialOperator );
      Devices::Cuda::freeFromDevice( kernelBoundaryConditions );
      Devices::Cuda::freeFromDevice( kernelRightHandSide );
      Devices::Cuda::freeFromDevice( kernelU );
      Devices::Cuda::freeFromDevice( kernelB );
      Devices::Cuda::freeFromDevice( kernelMatrix );
      checkCudaDevice;
   }*/
}

} // namespace TNL
