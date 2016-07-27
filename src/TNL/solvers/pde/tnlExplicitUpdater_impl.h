/***************************************************************************
                          tnlExplicitUpdater_impl.h  -  description
                             -------------------
    begin                : Jul 29, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>
#include <TNL/mesh/grids/tnlTraverser_Grid1D.h>
#include <TNL/mesh/grids/tnlTraverser_Grid2D.h>
#include <TNL/mesh/grids/tnlTraverser_Grid3D.h>

#include "tnlExplicitUpdater.h"


namespace TNL {

template< typename Mesh,
          typename MeshFunction,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide >
   template< typename EntityType >
void
tnlExplicitUpdater< Mesh, MeshFunction, DifferentialOperator, BoundaryConditions, RightHandSide >::
update( const RealType& time,
        const Mesh& mesh,
        const DifferentialOperator& differentialOperator,
        const BoundaryConditions& boundaryConditions,
        const RightHandSide& rightHandSide,
        MeshFunction& u,
        MeshFunction& fu ) const
{
   static_assert( std::is_same< MeshFunction,
                                Vectors::Vector< typename MeshFunction::RealType,
                                           typename MeshFunction::DeviceType,
                                           typename MeshFunction::IndexType > >::value != true,
      "Error: I am getting Vector instead of tnlMeshFunction or similar object. You might forget to bind DofVector into tnlMeshFunction in you method getExplicitRHS."  );
   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      TraverserUserData userData( time, differentialOperator, boundaryConditions, rightHandSide, u, fu );
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
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
      if( this->gpuTransferTimer )
         this->gpuTransferTimer->start();
      RealType* kernelTime = Devices::Cuda::passToDevice( time );
      DifferentialOperator* kernelDifferentialOperator = Devices::Cuda::passToDevice( differentialOperator );
      BoundaryConditions* kernelBoundaryConditions = Devices::Cuda::passToDevice( boundaryConditions );
      RightHandSide* kernelRightHandSide = Devices::Cuda::passToDevice( rightHandSide );
      MeshFunction* kernelU = Devices::Cuda::passToDevice( u );
      MeshFunction* kernelFu = Devices::Cuda::passToDevice( fu );
     if( this->gpuTransferTimer )
         this->gpuTransferTimer->stop();

      TraverserUserData userData( *kernelTime, *kernelDifferentialOperator, *kernelBoundaryConditions, *kernelRightHandSide, *kernelU, *kernelFu );
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

      if( this->gpuTransferTimer )
         this->gpuTransferTimer->start();
 
      checkCudaDevice;
      Devices::Cuda::freeFromDevice( kernelTime );
      Devices::Cuda::freeFromDevice( kernelDifferentialOperator );
      Devices::Cuda::freeFromDevice( kernelBoundaryConditions );
      Devices::Cuda::freeFromDevice( kernelRightHandSide );
      Devices::Cuda::freeFromDevice( kernelU );
      Devices::Cuda::freeFromDevice( kernelFu );
      checkCudaDevice;
 
      if( this->gpuTransferTimer )
         this->gpuTransferTimer->stop();

   }
}

} // namespace TNL
