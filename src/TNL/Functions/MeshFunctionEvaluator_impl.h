/***************************************************************************
                          MeshFunctionEvaluator.h  -  description
                             -------------------
    begin                : Jan 5, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Functions/MeshFunctionEvaluator.h>
#include <TNL/mesh/tnlTraverser.h>

namespace TNL {
namespace Functions {   

template< typename OutMeshFunction,
          typename InFunction >
void
MeshFunctionEvaluator< OutMeshFunction, InFunction >::
evaluate( OutMeshFunction& meshFunction,
          const InFunction& function,
          const RealType& time,
          const RealType& outFunctionMultiplicator,
          const RealType& inFunctionMultiplicator )
{
   switch( InFunction::getDomainType() )
   {
      case SpaceDomain:
      case MeshDomain:
         evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, all );
         break;
      case MeshInteriorDomain:
         evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, interior );
         break;
      case MeshBoundaryDomain:
         evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, boundary );
         break;
   }
}


template< typename OutMeshFunction,
          typename InFunction >
void
MeshFunctionEvaluator< OutMeshFunction, InFunction >::
evaluateAllEntities( OutMeshFunction& meshFunction,
                     const InFunction& function,
                     const RealType& time,
                     const RealType& outFunctionMultiplicator,
                     const RealType& inFunctionMultiplicator )
{
   return evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, all );
}

template< typename OutMeshFunction,
          typename InFunction >
void
MeshFunctionEvaluator< OutMeshFunction, InFunction >::
evaluateInteriorEntities( OutMeshFunction& meshFunction,
                          const InFunction& function,
                          const RealType& time,
                          const RealType& outFunctionMultiplicator,
                          const RealType& inFunctionMultiplicator )
{
   return evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, interior );
}

template< typename OutMeshFunction,
          typename InFunction >
void
MeshFunctionEvaluator< OutMeshFunction, InFunction >::
evaluateBoundaryEntities( OutMeshFunction& meshFunction,
                          const InFunction& function,
                          const RealType& time,
                          const RealType& outFunctionMultiplicator,
                          const RealType& inFunctionMultiplicator )
{
   return evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, boundary );
}



template< typename OutMeshFunction,
          typename InFunction >
void
MeshFunctionEvaluator< OutMeshFunction, InFunction >::
evaluateEntities( OutMeshFunction& meshFunction,
                  const InFunction& function,
                  const RealType& time,
                  const RealType& outFunctionMultiplicator,
                  const RealType& inFunctionMultiplicator,
                  EntitiesType entitiesType )
{
   typedef typename MeshType::template MeshEntity< OutMeshFunction::getEntitiesDimensions() > MeshEntityType;
   typedef Functions::MeshFunctionEvaluatorAssignmentEntitiesProcessor< MeshType, TraverserUserData > AssignmentEntitiesProcessor;
   typedef Functions::MeshFunctionEvaluatorAdditionEntitiesProcessor< MeshType, TraverserUserData > AdditionEntitiesProcessor;
 
   if( std::is_same< MeshDeviceType, Devices::Host >::value )
   {
      TraverserUserData userData( &function, &time, &meshFunction, &outFunctionMultiplicator, &inFunctionMultiplicator );
      tnlTraverser< MeshType, MeshEntityType > meshTraverser;
      switch( entitiesType )
      {
         case all:
            if( outFunctionMultiplicator )
               meshTraverser.template processAllEntities< TraverserUserData,
                                                          AdditionEntitiesProcessor >
                                                        ( meshFunction.getMeshPointer(),
                                                          userData );
            else
               meshTraverser.template processAllEntities< TraverserUserData,
                                                         AssignmentEntitiesProcessor >
                                                       ( meshFunction.getMeshPointer(),
                                                         userData );
            break;
         case interior:
            if( outFunctionMultiplicator )
               meshTraverser.template processInteriorEntities< TraverserUserData,
                                                               AdditionEntitiesProcessor >
                                                             ( meshFunction.getMeshPointer(),
                                                               userData );
            else
               meshTraverser.template processInteriorEntities< TraverserUserData,
                                                               AssignmentEntitiesProcessor >
                                                             ( meshFunction.getMeshPointer(),
                                                               userData );            
            break;
         case boundary:
            if( outFunctionMultiplicator )
               meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                               AdditionEntitiesProcessor >
                                                             ( meshFunction.getMeshPointer(),
                                                               userData );
            else
               meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                               AssignmentEntitiesProcessor >
                                                             ( meshFunction.getMeshPointer(),
                                                               userData );
            break;
      }
   }
   if( std::is_same< MeshDeviceType, Devices::Cuda >::value )
   {
      OutMeshFunction* kernelMeshFunction = Devices::Cuda::passToDevice( meshFunction );
      InFunction* kernelFunction = Devices::Cuda::passToDevice( function );
      RealType* kernelTime = Devices::Cuda::passToDevice( time );
      RealType* kernelOutFunctionMultiplicator = Devices::Cuda::passToDevice( outFunctionMultiplicator );
      RealType* kernelInFunctionMultiplicator = Devices::Cuda::passToDevice( inFunctionMultiplicator );
 
      TraverserUserData userData( kernelFunction, kernelTime, kernelMeshFunction, kernelOutFunctionMultiplicator, kernelInFunctionMultiplicator );
      checkCudaDevice;
      tnlTraverser< MeshType, MeshEntityType > meshTraverser;
      switch( entitiesType )
      {
         case all:
            if( outFunctionMultiplicator )
               meshTraverser.template processAllEntities< TraverserUserData,
                                                          AdditionEntitiesProcessor >
                                                        ( meshFunction.getMeshPointer(),
                                                          userData );
            else
               meshTraverser.template processAllEntities< TraverserUserData,
                                                         AssignmentEntitiesProcessor >
                                                       ( meshFunction.getMeshPointer(),
                                                         userData );
            break;
         case interior:
            if( outFunctionMultiplicator )
               meshTraverser.template processInteriorEntities< TraverserUserData,
                                                               AdditionEntitiesProcessor >
                                                             ( meshFunction.getMeshPointer(),
                                                               userData );
            else
               meshTraverser.template processInteriorEntities< TraverserUserData,
                                                               AssignmentEntitiesProcessor >
                                                             ( meshFunction.getMeshPointer(),
                                                               userData );            
            break;
         case boundary:
            if( outFunctionMultiplicator )
               meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                               AdditionEntitiesProcessor >
                                                             ( meshFunction.getMeshPointer(),
                                                               userData );
            else
               meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                               AssignmentEntitiesProcessor >
                                                             ( meshFunction.getMeshPointer(),
                                                               userData );
            break;
      }

      checkCudaDevice;
      Devices::Cuda::freeFromDevice( kernelMeshFunction );
      Devices::Cuda::freeFromDevice( kernelFunction );
      Devices::Cuda::freeFromDevice( kernelTime );
      Devices::Cuda::freeFromDevice( kernelOutFunctionMultiplicator );
      Devices::Cuda::freeFromDevice( kernelInFunctionMultiplicator );
      checkCudaDevice;
   }
}

} // namespace Functions
} // namespace TNL

