/***************************************************************************
                          tnlFunctionEnumerator_impl.h  -  description
                             -------------------
    begin                : Mar 5, 2015
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
#ifndef SRC_FUNCTIONS_TNLFUNCTIONENUMERATOR_IMPL_H_
#define SRC_FUNCTIONS_TNLFUNCTIONENUMERATOR_IMPL_H_

#include <functors/tnlFunctionEnumerator.h>
#include <mesh/grids/tnlTraverser_Grid1D.h>
#include <mesh/grids/tnlTraverser_Grid2D.h>
#include <mesh/grids/tnlTraverser_Grid3D.h>

template< typename Mesh,
          typename Function,
          typename DofVector >
   template< int EntityDimensions >
void
tnlFunctionEnumerator< Mesh, Function, DofVector >::
enumerate( const MeshType& mesh,
           const Function& function,
           DofVector& u,
           const RealType& functionCoefficient,
           const RealType& dofVectorCoefficient,
           const RealType& time ) const

{
   if( DeviceType::DeviceType == tnlHostDevice )
   {
      TraverserUserData userData( time, function, u, functionCoefficient, dofVectorCoefficient );
      tnlTraverser< MeshType, EntityDimensions > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
      meshTraverser.template processInteriorEntities< TraverserUserData,
                                                      TraverserEntitiesProcessor >
                                                    ( mesh,
                                                      userData );

   }
   if( DeviceType::DeviceType == tnlCudaDevice )
   {
      RealType* kernelTime = tnlCuda::passToDevice( time );
      Function* kernelFunction = tnlCuda::passToDevice( function );
      DofVector* kernelU = tnlCuda::passToDevice( u );
      RealType* kernelFunctionCoefficient = tnlCuda::passToDevice( functionCoefficient );
      RealType* kernelDofVectorCoefficient = tnlCuda::passToDevice( dofVectorCoefficient );
      TraverserUserData userData( *kernelTime, *kernelFunction, *kernelU, *kernelFunctionCoefficient, *kernelDofVectorCoefficient );
      checkCudaDevice;
      tnlTraverser< MeshType, EntityDimensions > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
      meshTraverser.template processInteriorEntities< TraverserUserData,
                                                      TraverserEntitiesProcessor >
                                                    ( mesh,
                                                      userData );

      checkCudaDevice;
      tnlCuda::freeFromDevice( kernelTime );
      tnlCuda::freeFromDevice( kernelFunction );
      tnlCuda::freeFromDevice( kernelU );
      tnlCuda::freeFromDevice( kernelFunctionCoefficient );
      tnlCuda::freeFromDevice( kernelDofVectorCoefficient );
      checkCudaDevice;
   }
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Function,
          typename DofVector >
   template< int EntityDimensions >
void
tnlFunctionEnumerator< tnlGrid< Dimensions, Real, Device, Index >, Function, DofVector  >::
enumerate( const tnlGrid< Dimensions, Real, Device, Index >& mesh,
           const Function& function,
           DofVector& u,
           const RealType& functionCoefficient,
           const RealType& dofVectorCoefficient,
           const RealType& time ) const
{
   if( ( tnlDeviceEnum ) DeviceType::DeviceType == tnlHostDevice )
   {
      TraverserUserData userData( time, function, u, functionCoefficient, dofVectorCoefficient );
      tnlTraverser< MeshType, EntityDimensions > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
      meshTraverser.template processInteriorEntities< TraverserUserData,
                                                      TraverserEntitiesProcessor >
                                                    ( mesh,
                                                      userData );

   }
   if( ( tnlDeviceEnum ) DeviceType::DeviceType == tnlCudaDevice )
   {
      RealType* kernelTime = tnlCuda::passToDevice( time );
      Function* kernelFunction = tnlCuda::passToDevice( function );
      DofVector* kernelU = tnlCuda::passToDevice( u );
      RealType* kernelFunctionCoefficient = tnlCuda::passToDevice( functionCoefficient );
      RealType* kernelDofVectorCoefficient = tnlCuda::passToDevice( dofVectorCoefficient );
      TraverserUserData userData( *kernelTime, *kernelFunction, *kernelU, *kernelFunctionCoefficient, *kernelDofVectorCoefficient );
      checkCudaDevice;
      tnlTraverser< MeshType, EntityDimensions > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
      meshTraverser.template processInteriorEntities< TraverserUserData,
                                                      TraverserEntitiesProcessor >
                                                    ( mesh,
                                                      userData );

      checkCudaDevice;
      tnlCuda::freeFromDevice( kernelTime );
      tnlCuda::freeFromDevice( kernelFunction );
      tnlCuda::freeFromDevice( kernelU );
      tnlCuda::freeFromDevice( kernelFunctionCoefficient );
      tnlCuda::freeFromDevice( kernelDofVectorCoefficient );
      checkCudaDevice;
   }
}



#endif /* SRC_FUNCTIONS_TNLFUNCTIONENUMERATOR_IMPL_H_ */
