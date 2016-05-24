/***************************************************************************
                          tnlBoundaryConditionsSetter_impl.h  -  description
                             -------------------
    begin                : Dec 30, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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

#ifndef TNLBOUNDARYCONDITIONSSETTER_IMPL_H
#define	TNLBOUNDARYCONDITIONSSETTER_IMPL_H

#include <type_traits>

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
   
   if( std::is_same< DeviceType, tnlMIC >::value )
   {
      const RealType* kernelTime = tnlMIC::passToDevice( time );
      const BoundaryConditions* kernelBoundaryConditions = tnlMIC::passToDevice( boundaryConditions );
      MeshFunction* kernelU = tnlMIC::passToDevice( u );
      TraverserUserData userData( *kernelTime, *kernelBoundaryConditions, *kernelU );
     
      tnlTraverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserBoundaryEntitiesProcessor >
                                                    ( u.getMesh(),
                                                      userData );
      tnlMIC::freeFromDevice( kernelTime );
      tnlMIC::freeFromDevice( kernelBoundaryConditions );
      tnlMIC::freeFromDevice( kernelU );
     
   }
   
}



#endif	/* TNLBOUNDARYCONDITIONSSETTER_IMPL_H */

