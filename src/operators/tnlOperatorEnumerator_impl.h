/***************************************************************************
                          tnlOperatorEnumerator_impl.h  -  description
                             -------------------
    begin                : Mar 8, 2015
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
#ifndef SRC_OPERATORS_TNLOPERATORENUMERATOR_IMPL_H_
#define SRC_OPERATORS_TNLOPERATORENUMERATOR_IMPL_H_

#include <operators/tnlOperatorEnumerator.h>
#include <mesh/tnlTraverser_Grid1D.h>
#include <mesh/tnlTraverser_Grid2D.h>
#include <mesh/tnlTraverser_Grid3D.h>

template< typename Mesh,
          typename Operator,
          typename DofVector >
   template< int EntityDimensions >
void
tnlOperatorEnumerator< Mesh, Operator, DofVector >::
enumerate( const MeshType& mesh,
           const Operator& _operator,
           DofVector& u,
           const RealType& _operatorCoefficient,
           const RealType& dofVectorCoefficient,
           const RealType& time ) const

{
   if( DeviceType::DeviceType == tnlHostDevice )
   {
      TraverserUserData userData( time, _operator, u, _operatorCoefficient, dofVectorCoefficient );
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
      Operator* kernelOperator = tnlCuda::passToDevice( _operator );
      DofVector* kernelU = tnlCuda::passToDevice( u );
      RealType* kernelOperatorCoefficient = tnlCuda::passToDevice( _operatorCoefficient );
      RealType* kernelDofVectorCoefficient = tnlCuda::passToDevice( dofVectorCoefficient );
      TraverserUserData userData( *kernelTime, *kernelOperator, *kernelU, *kernelOperatorCoefficient, *kernelDofVectorCoefficient );
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
      tnlCuda::freeFromDevice( kernelOperator );
      tnlCuda::freeFromDevice( kernelU );
      tnlCuda::freeFromDevice( kernelOperatorCoefficient );
      tnlCuda::freeFromDevice( kernelDofVectorCoefficient );
      checkCudaDevice;
   }
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Operator,
          typename DofVector >
   template< int EntityDimensions >
void
tnlOperatorEnumerator< tnlGrid< Dimensions, Real, Device, Index >, Operator, DofVector  >::
enumerate( const tnlGrid< Dimensions, Real, Device, Index >& mesh,
           const Operator& _operator,
           DofVector& u,
           const RealType& _operatorCoefficient,
           const RealType& dofVectorCoefficient,
           const RealType& time ) const
{
   if( DeviceType::DeviceType == tnlHostDevice )
   {
      TraverserUserData userData( time, _operator, u, _operatorCoefficient, dofVectorCoefficient );
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
      Operator* kernelOperator = tnlCuda::passToDevice( _operator );
      DofVector* kernelU = tnlCuda::passToDevice( u );
      RealType* kernelOperatorCoefficient = tnlCuda::passToDevice( _operatorCoefficient );
      RealType* kernelDofVectorCoefficient = tnlCuda::passToDevice( dofVectorCoefficient );
      TraverserUserData userData( *kernelTime, *kernelOperator, *kernelU, *kernelOperatorCoefficient, *kernelDofVectorCoefficient );
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
      tnlCuda::freeFromDevice( kernelOperator );
      tnlCuda::freeFromDevice( kernelU );
      tnlCuda::freeFromDevice( kernelOperatorCoefficient );
      tnlCuda::freeFromDevice( kernelDofVectorCoefficient );
      checkCudaDevice;
   }
}

#endif /* SRC_OPERATORS_TNLOPERATORENUMERATOR_IMPL_H_ */
