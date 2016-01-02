/***************************************************************************
                          tnlFunctionEvaluator_impl.h  -  description
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
#ifndef SRC_FUNCTIONS_TNLFUNCTIONEVALUATOR_IMPL_H_
#define SRC_FUNCTIONS_TNLFUNCTIONEVALUATOR_IMPL_H_

#include <type_traits>
#include <functions/tnlFunctionEvaluator.h>
#include <mesh/grids/tnlTraverser_Grid1D.h>
#include <mesh/grids/tnlTraverser_Grid2D.h>
#include <mesh/grids/tnlTraverser_Grid3D.h>

template< typename MeshFunction,
          typename Function >
void
tnlFunctionEvaluator< MeshFunction, Function >::
assignment( const Function& function,
            MeshFunction& u,
            const RealType& functionCoefficient,
            const RealType& dofVectorCoefficient,
            const RealType& time ) const

{
   typedef typename MeshType::template MeshEntity< MeshFunction::EntityDimensions > MeshEntityType;
   if( std::is_same< DeviceType, tnlHost >::value )
   {
      TraverserUserData userData( time, function, u, functionCoefficient, dofVectorCoefficient );
      tnlTraverser< MeshType, MeshEntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserEntitiesProcessor >
                                                    ( u.getMesh(),
                                                      userData );
      meshTraverser.template processInteriorEntities< TraverserUserData,
                                                      TraverserEntitiesProcessor >
                                                    ( u.getMesh(),
                                                      userData );

   }
   if( std::is_same< DeviceType, tnlCuda >::value )
   {
      RealType* kernelTime = tnlCuda::passToDevice( time );
      Function* kernelFunction = tnlCuda::passToDevice( function );
      MeshFunction* kernelU = tnlCuda::passToDevice( u );
      RealType* kernelFunctionCoefficient = tnlCuda::passToDevice( functionCoefficient );
      RealType* kernelDofVectorCoefficient = tnlCuda::passToDevice( dofVectorCoefficient );
      TraverserUserData userData( *kernelTime, *kernelFunction, *kernelU, *kernelFunctionCoefficient, *kernelDofVectorCoefficient );
      checkCudaDevice;
      tnlTraverser< MeshType, MeshEntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserEntitiesProcessor >
                                                    ( u.getMesh(),
                                                      userData );
      meshTraverser.template processInteriorEntities< TraverserUserData,
                                                      TraverserEntitiesProcessor >
                                                    ( u.getMesh(),
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

#endif /* SRC_FUNCTIONS_TNLFUNCTIONEVALUATOR_IMPL_H_ */
