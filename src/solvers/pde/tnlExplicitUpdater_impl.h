/***************************************************************************
                          tnlExplicitUpdater_impl.h  -  description
                             -------------------
    begin                : Jul 29, 2014
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

#ifndef TNLEXPLICITUPDATER_IMPL_H_
#define TNLEXPLICITUPDATER_IMPL_H_

#include <type_traits>
#include <mesh/grids/tnlTraverser_Grid1D.h>
#include <mesh/grids/tnlTraverser_Grid2D.h>
#include <mesh/grids/tnlTraverser_Grid3D.h>

#include <execinfo.h>
#include <stdint.h>

#include "tnlExplicitUpdater.h"

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
                                tnlVector< typename MeshFunction::RealType,
                                           typename MeshFunction::DeviceType,
                                           typename MeshFunction::IndexType > >::value != true,
      "Error: I am getting tnlVector instead of tnlMeshFunction or similar object. You might forget to bind DofVector into tnlMeshFunction in you method getExplicitRHS."  );
   //HOST
   if( std::is_same< DeviceType, tnlHost >::value )
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
   //CUDA
   if( std::is_same< DeviceType, tnlCuda >::value )
   {
      if( this->gpuTransferTimer ) 
         this->gpuTransferTimer->start();
      RealType* kernelTime = tnlCuda::passToDevice( time );
      DifferentialOperator* kernelDifferentialOperator = tnlCuda::passToDevice( differentialOperator );
      BoundaryConditions* kernelBoundaryConditions = tnlCuda::passToDevice( boundaryConditions );
      RightHandSide* kernelRightHandSide = tnlCuda::passToDevice( rightHandSide );
      MeshFunction* kernelU = tnlCuda::passToDevice( u );
      MeshFunction* kernelFu = tnlCuda::passToDevice( fu );
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
      tnlCuda::freeFromDevice( kernelTime );
      tnlCuda::freeFromDevice( kernelDifferentialOperator );
      tnlCuda::freeFromDevice( kernelBoundaryConditions );
      tnlCuda::freeFromDevice( kernelRightHandSide );
      tnlCuda::freeFromDevice( kernelU );
      tnlCuda::freeFromDevice( kernelFu );
      checkCudaDevice;
      
      if( this->gpuTransferTimer ) 
         this->gpuTransferTimer->stop();

   }
   //MIC
   if( std::is_same< DeviceType, tnlMIC >::value )
   {  
      /*
      //Like CUDA
      const RealType* kernelTime = tnlMIC::passToDevice( time );
      const DifferentialOperator* kernelDifferentialOperator = tnlMIC::passToDevice( differentialOperator );
      const BoundaryConditions* kernelBoundaryConditions = tnlMIC::passToDevice( boundaryConditions );
      const RightHandSide* kernelRightHandSide = tnlMIC::passToDevice( rightHandSide );
      MeshFunction* kernelU = tnlMIC::passToDevice( u );
      MeshFunction* kernelFu = tnlMIC::passToDevice( fu );*/
            
       
#define USE_MICSTRUCT
       
//nejrychlejsi varianta
#ifdef USE_MICSTRUCT    
       TraverserUserData userData( time, differentialOperator, boundaryConditions, rightHandSide, u, fu );
       
       TNLMICSTRUCT(time,const RealType);
       TNLMICSTRUCT(differentialOperator,const DifferentialOperator);
       TNLMICSTRUCT(boundaryConditions,const BoundaryConditions);
       TNLMICSTRUCT(rightHandSide,const RightHandSide);
       TNLMICSTRUCT(u,MeshFunction);
       TNLMICSTRUCT(fu,MeshFunction);
       
       TNLMICSTRUCT(userData, TraverserUserData);
       
      // cout << "userData:" << sizeof(userData) << endl;
      // cout << "suserData:" << sizeof(suserData) << endl;
       
       #pragma offload target(mic) in(stime,sdifferentialOperator,sboundaryConditions,srightHandSide,su,sfu) inout(suserData)
       {
           TNLMICSTRUCTUSE(userData,TraverserUserData);
           
           TNLMICSTRUCTALLOC(time,const RealType);
           TNLMICSTRUCTALLOC(differentialOperator,const DifferentialOperator);
           TNLMICSTRUCTALLOC(boundaryConditions,const BoundaryConditions);
           TNLMICSTRUCTALLOC(rightHandSide,const RightHandSide);
           TNLMICSTRUCTALLOC(u,MeshFunction);
           TNLMICSTRUCTALLOC(fu,MeshFunction);
           
           
           kerneluserData->time=kerneltime;
           kerneluserData->differentialOperator=kerneldifferentialOperator;
           kerneluserData->boundaryConditions=kernelboundaryConditions;
           kerneluserData->rightHandSide=kernelrightHandSide;
           kerneluserData->u=kernelu;
           kerneluserData->fu=kernelfu;
           
       }
       
       memcpy((void*)&userData,(void*)&suserData,sizeof(TraverserUserData));
       
#endif
       
      
#ifdef USE_MICHIDE
     
       TNLMICHIDE(time,const RealType);
       TNLMICHIDE(differentialOperator,const DifferentialOperator);
       TNLMICHIDE(boundaryConditions,const BoundaryConditions);
       TNLMICHIDE(rightHandSide,const RightHandSide);
       TNLMICHIDE(u,MeshFunction);
       TNLMICHIDE(fu,MeshFunction);
       
//#pragma offload target(mic) in(utime:length(sizeof(RealType))) in(udifferentialOperator:length(sizeof(DifferentialOperator))) in(uboundaryConditions:length(sizeof(BoundaryConditions))) in(urightHandSide:length(sizeof(RightHandSide))) in(uu:length(sizeof(MeshFunction))) in(ufu:length(sizeof(MeshFunction))) out(kernelTime,kernelDifferentialOperator,kernelBoundaryConditions,kernelRightHandSide,kernelU, kernelFu)
#pragma offload target(mic) TNLMICHIDEALLOCOFF(time,const RealType) TNLMICHIDEALLOCOFF(differentialOperator,const DifferentialOperator) TNLMICHIDEALLOCOFF(boundaryConditions,const BoundaryConditions) TNLMICHIDEALLOCOFF(rightHandSide,const RightHandSide) TNLMICHIDEALLOCOFF(u,MeshFunction) TNLMICHIDEALLOCOFF(fu,MeshFunction)
{
           
            TNLMICHIDEALLOC(time,const RealType);
            TNLMICHIDEALLOC(differentialOperator,const DifferentialOperator);
            TNLMICHIDEALLOC(boundaryConditions,const BoundaryConditions);
            TNLMICHIDEALLOC(rightHandSide,const RightHandSide);
            TNLMICHIDEALLOC(u,MeshFunction);
            TNLMICHIDEALLOC(fu,MeshFunction);
}        
    
       TraverserUserData userData( *kerneltime.pointer, *kerneldifferentialOperator.pointer, *kernelboundaryConditions.pointer, *kernelrightHandSide.pointer, *kernelu.pointer, *kernelfu.pointer );
           
#endif
       
      tnlTraverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserBoundaryEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
      meshTraverser.template processInteriorEntities< TraverserUserData,
                                                      TraverserInteriorEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
      
      /*
       Like CUDA
      tnlMIC::freeFromDevice( kernelTime );
      tnlMIC::freeFromDevice( kernelDifferentialOperator );
      tnlMIC::freeFromDevice( kernelBoundaryConditions );
      tnlMIC::freeFromDevice( kernelRightHandSide );
      tnlMIC::freeFromDevice( kernelU );
      tnlMIC::freeFromDevice( kernelFu );       */
      

    
#ifdef USE_MICSTRUCT
        #pragma offload target(mic) in(suserData)
        {      
          TNLMICSTRUCTUSE(userData,TraverserUserData);
           free((void*)kerneluserData->time);
           free((void*)kerneluserData->differentialOperator);
           free((void*)kerneluserData->boundaryConditions);
           free((void*)kerneluserData->rightHandSide);
           free((void*)kerneluserData->u);
           free((void*)kerneluserData->fu);
              
        }           
#endif
#ifdef USE_MICHIDE                                               
        #pragma offload target(mic) TNLMICHIDEFREEOFF(time,const RealType) TNLMICHIDEFREEOFF(differentialOperator,const DifferentialOperator) TNLMICHIDEFREEOFF(boundaryConditions,const BoundaryConditions) TNLMICHIDEFREEOFF(rightHandSide,const RightHandSide) TNLMICHIDEFREEOFF(u,MeshFunction) TNLMICHIDEFREEOFF(fu,MeshFunction)
       {
            TNLMICHIDEFREE(time,const RealType);
            TNLMICHIDEFREE(differentialOperator,const DifferentialOperator);
            TNLMICHIDEFREE(boundaryConditions,const BoundaryConditions);
            TNLMICHIDEFREE(rightHandSide,const RightHandSide);
            TNLMICHIDEFREE(u,MeshFunction);
            TNLMICHIDEFREE(fu,MeshFunction);
        }
#endif
   }
}

#endif /* TNLEXPLICITUPDATER_IMPL_H_ */
