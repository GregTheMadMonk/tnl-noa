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

namespace TNL {

template< typename MeshFunction,
          typename BoundaryConditions >
   template< typename EntityType >
void
tnlBoundaryConditionsSetter< MeshFunction, BoundaryConditions >::
apply( const BoundaryConditions& boundaryConditions,
       const RealType& time,
       MeshFunction& u )
{
   if( std::is_same< DeviceType, tnlHost >::value )
   {
      TraverserUserData userData( time, boundaryConditions, u );
      tnlTraverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserBoundaryEntitiesProcessor >
                                                    ( u.getMesh(),
                                                      userData );
   }
   if( std::is_same< DeviceType, tnlCuda >::value )
   {
      RealType* kernelTime = tnlCuda::passToDevice( time );
      BoundaryConditions* kernelBoundaryConditions = tnlCuda::passToDevice( boundaryConditions );
      MeshFunction* kernelU = tnlCuda::passToDevice( u );
      TraverserUserData userData( *kernelTime, *kernelBoundaryConditions, *kernelU );
      checkCudaDevice;
      tnlTraverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserBoundaryEntitiesProcessor >
                                                    ( u.getMesh(),
                                                      userData );
      tnlCuda::freeFromDevice( kernelTime );
      tnlCuda::freeFromDevice( kernelBoundaryConditions );
      tnlCuda::freeFromDevice( kernelU );
      checkCudaDevice;
   }
}

} // namespace TNL

