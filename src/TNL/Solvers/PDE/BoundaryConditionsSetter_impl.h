/***************************************************************************
                          BoundaryConditionsSetter_impl.h  -  description
                             -------------------
    begin                : Dec 30, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>
#include <TNL/Meshes/Traverser.h>

namespace TNL {
namespace Solvers {
namespace PDE {   

template< typename MeshFunction,
          typename BoundaryConditions >
   template< typename EntityType >
void
BoundaryConditionsSetter< MeshFunction, BoundaryConditions >::
apply( const BoundaryConditionsPointer& boundaryConditions,
       const RealType& time,
       MeshFunctionPointer& u )
{
   //if( std::is_same< DeviceType, Devices::Host >::value )
   {
      TraverserUserData userData( time,
                                  boundaryConditions.template getData< DeviceType >(),
                                  u.template modifyData< DeviceType >() );
      Meshes::Traverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserBoundaryEntitiesProcessor >
                                                    ( u->getMeshPointer(),
                                                      userData );
   }
   /*if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
      RealType* kernelTime = Devices::Cuda::passToDevice( time );
      BoundaryConditions* kernelBoundaryConditions = Devices::Cuda::passToDevice( boundaryConditions );
      MeshFunction* kernelU = Devices::Cuda::passToDevice( u );
      TraverserUserData userData( *kernelTime, *kernelBoundaryConditions, *kernelU );
      checkCudaDevice;
      Meshes::Traverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserBoundaryEntitiesProcessor >
                                                    ( u.getMeshPointer(),
                                                      userData );
      Devices::Cuda::freeFromDevice( kernelTime );
      Devices::Cuda::freeFromDevice( kernelBoundaryConditions );
      Devices::Cuda::freeFromDevice( kernelU );
      checkCudaDevice;
   }*/
}

} // namespace PDE
} // namespace Solvers
} // namespace TNL

