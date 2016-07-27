/***************************************************************************
                          tnlFunctionEvaluator_impl.h  -  description
                             -------------------
    begin                : Mar 5, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>
#include <TNL/functions/tnlFunctionEvaluator.h>
#include <TNL/mesh/grids/tnlTraverser_Grid1D.h>
#include <TNL/mesh/grids/tnlTraverser_Grid2D.h>
#include <TNL/mesh/grids/tnlTraverser_Grid3D.h>

namespace TNL {

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
   if( std::is_same< DeviceType, Devices::Host >::value )
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
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
      RealType* kernelTime = Devices::Cuda::passToDevice( time );
      Function* kernelFunction = Devices::Cuda::passToDevice( function );
      MeshFunction* kernelU = Devices::Cuda::passToDevice( u );
      RealType* kernelFunctionCoefficient = Devices::Cuda::passToDevice( functionCoefficient );
      RealType* kernelDofVectorCoefficient = Devices::Cuda::passToDevice( dofVectorCoefficient );
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
      Devices::Cuda::freeFromDevice( kernelTime );
      Devices::Cuda::freeFromDevice( kernelFunction );
      Devices::Cuda::freeFromDevice( kernelU );
      Devices::Cuda::freeFromDevice( kernelFunctionCoefficient );
      Devices::Cuda::freeFromDevice( kernelDofVectorCoefficient );
      checkCudaDevice;
   }
}

} // namespace TNL
