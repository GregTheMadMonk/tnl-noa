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
#include <core/tnlMIC.h>

#include "tnlBoundaryConditionsSetter.h"

#include <stdint.h>

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
      
      /*const RealType* kernelTime = tnlMIC::passToDevice( time );
      const BoundaryConditions* kernelBoundaryConditions = tnlMIC::passToDevice( boundaryConditions );
      MeshFunction* kernelU = tnlMIC::passToDevice( u );
      TraverserUserData userData( *kernelTime, *kernelBoundaryConditions, *kernelU );
       */
       
#define USE_MICSTRUCT
#ifdef USE_MICSTRUCT
       TraverserUserData userData( time, boundaryConditions, u );
       
       TNLMICSTRUCT(time,const RealType);
       TNLMICSTRUCT(boundaryConditions,const BoundaryConditions);
       TNLMICSTRUCT(u,MeshFunction);
       
       TNLMICSTRUCT(userData,TraverserUserData);
       
       #pragma offload target(mic) in(stime,sboundaryConditions,su) inout(suserData)
       {
            TNLMICSTRUCTALLOC(time,const RealType);
            TNLMICSTRUCTALLOC(boundaryConditions,const BoundaryConditions);
            TNLMICSTRUCTALLOC(u,MeshFunction);
            
            TNLMICSTRUCTUSE(userData,TraverserUserData);
            
            kerneluserData->boundaryConditions=kernelboundaryConditions;
            kerneluserData->time=kerneltime;
            kerneluserData->u=kernelu;     
       }
       memcpy((void*)&userData,(void*)&suserData,sizeof(TraverserUserData));
#endif
       
#ifdef USE_MICHIDE
      uint8_t * utime=(uint8_t*)&time;
      satanHider<RealType> kernelTime;
      uint8_t * uboundaryConditions=(uint8_t*)&boundaryConditions;
      satanHider<BoundaryConditions> kernelBoundaryConditions;
      uint8_t * uu=(uint8_t*)&u;
      satanHider<MeshFunction> kernelU;

#pragma offload target(mic) in(utime:length(sizeof(RealType))) in(uboundaryConditions:length(sizeof(BoundaryConditions))) in(uu:length(sizeof(MeshFunction)))
{
    kernelTime.pointer=(RealType*)malloc(sizeof(RealType));
    memcpy((void*)kernelTime.pointer,(void*)utime,sizeof(RealType));   
    kernelBoundaryConditions.pointer=(BoundaryConditions*)malloc(sizeof(BoundaryConditions));
    memcpy((void*)kernelBoundaryConditions.pointer,(void*)uboundaryConditions,sizeof(BoundaryConditions));
    kernelU.pointer=(MeshFunction*)malloc(sizeof(MeshFunction));
    memcpy((void*)kernelU.pointer,(void*)uu,sizeof(MeshFunction));      
}   
      TraverserUserData userData( *kernelTime.pointer, *kernelBoundaryConditions.pointer, *kernelU.pointer );
#endif    
      
      
      tnlTraverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserBoundaryEntitiesProcessor >
                                                    ( u.getMesh(),
                                                      userData );
      /*
      tnlMIC::freeFromDevice( kernelTime );
      tnlMIC::freeFromDevice( kernelBoundaryConditions );
      tnlMIC::freeFromDevice( kernelU );
     */
    
#ifdef USE_MICSTRUCT
    #pragma offload target(mic) in(suserData)
    {
         TNLMICSTRUCTUSE(userData,TraverserUserData);
         
         free((void*)kerneluserData->boundaryConditions);
         free((void*)kerneluserData->time);
         free((void*)kerneluserData->u);
    }
#endif

#ifdef USE_MICHIDE
#pragma offload target(mic) in(kernelTime,kernelBoundaryConditions,kernelU)
      {
          free((void*)kernelTime.pointer);
          free((void*)kernelBoundaryConditions.pointer);
          free((void*)kernelU.pointer);
      }
#endif
       
                                                      
   }
   
}



#endif	/* TNLBOUNDARYCONDITIONSSETTER_IMPL_H */

