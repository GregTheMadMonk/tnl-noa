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
      
      const RealType* kernelTime = tnlMIC::passToDevice( time );
      const DifferentialOperator* kernelDifferentialOperator = tnlMIC::passToDevice( differentialOperator );
      const BoundaryConditions* kernelBoundaryConditions = tnlMIC::passToDevice( boundaryConditions );
      const RightHandSide* kernelRightHandSide = tnlMIC::passToDevice( rightHandSide );
      MeshFunction* kernelU = tnlMIC::passToDevice( u );
      MeshFunction* kernelFu = tnlMIC::passToDevice( fu );
           
      TraverserUserData userData( *kernelTime, *kernelDifferentialOperator, *kernelBoundaryConditions, *kernelRightHandSide, *kernelU, *kernelFu );
       
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
      tnlMIC::freeFromDevice( kernelDifferentialOperator );
      tnlMIC::freeFromDevice( kernelBoundaryConditions );
      tnlMIC::freeFromDevice( kernelRightHandSide );
      tnlMIC::freeFromDevice( kernelU );
      tnlMIC::freeFromDevice( kernelFu );       
   }
}

#endif /* TNLEXPLICITUPDATER_IMPL_H_ */
