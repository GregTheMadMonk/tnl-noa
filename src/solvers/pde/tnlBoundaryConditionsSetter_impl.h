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

template< typename Mesh,
          typename DofVector,
          typename BoundaryConditions >
   template< typename EntityType >
void
tnlBoundaryConditionsSetter< Mesh, DofVector, BoundaryConditions >::
apply( const RealType& time,
        const Mesh& mesh,
        const BoundaryConditions& boundaryConditions,
        DofVector& u ) const
{
   if( ( tnlDeviceEnum ) DeviceType::DeviceType == tnlHostDevice )
   {
      TraverserUserData userData( time, boundaryConditions, u );
      tnlTraverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserBoundaryEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
   }
   if( ( tnlDeviceEnum ) DeviceType::DeviceType == tnlCudaDevice )
   {
      RealType* kernelTime = tnlCuda::passToDevice( time );
      BoundaryConditions* kernelBoundaryConditions = tnlCuda::passToDevice( boundaryConditions );
      DofVector* kernelU = tnlCuda::passToDevice( u );
      TraverserUserData userData( *kernelTime, *kernelBoundaryConditions, *kernelU );
      checkCudaDevice;
      tnlTraverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserBoundaryEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
      tnlCuda::freeFromDevice( kernelTime );
      tnlCuda::freeFromDevice( kernelBoundaryConditions );
      tnlCuda::freeFromDevice( kernelU );
      checkCudaDevice;
   }
}



#endif	/* TNLBOUNDARYCONDITIONSSETTER_IMPL_H */

