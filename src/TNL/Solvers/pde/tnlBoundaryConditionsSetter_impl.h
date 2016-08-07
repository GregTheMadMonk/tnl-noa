/***************************************************************************
                          tnlBoundaryConditionsSetter_impl.h  -  description
                             -------------------
    begin                : Dec 30, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>
#include <TNL/mesh/tnlTraverser.h>

namespace TNL {
namespace Solvers {   

template< typename MeshFunction,
          typename BoundaryConditions >
   template< typename EntityType >
void
tnlBoundaryConditionsSetter< MeshFunction, BoundaryConditions >::
apply( const BoundaryConditions& boundaryConditions,
       const RealType& time,
       MeshFunction& u )
{
   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      TraverserUserData userData( time, boundaryConditions, u );
      tnlTraverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserBoundaryEntitiesProcessor >
                                                    ( u.getMeshPointer(),
                                                      userData );
   }
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
      RealType* kernelTime = Devices::Cuda::passToDevice( time );
      BoundaryConditions* kernelBoundaryConditions = Devices::Cuda::passToDevice( boundaryConditions );
      MeshFunction* kernelU = Devices::Cuda::passToDevice( u );
      TraverserUserData userData( *kernelTime, *kernelBoundaryConditions, *kernelU );
      checkCudaDevice;
      tnlTraverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserBoundaryEntitiesProcessor >
                                                    ( u.getMeshPointer(),
                                                      userData );
      Devices::Cuda::freeFromDevice( kernelTime );
      Devices::Cuda::freeFromDevice( kernelBoundaryConditions );
      Devices::Cuda::freeFromDevice( kernelU );
      checkCudaDevice;
   }
}

} // namespace Solvers
} // namespace TNL

