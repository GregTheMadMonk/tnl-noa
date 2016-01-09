/***************************************************************************
                          tnlMeshFunctionEvaluator.h  -  description
                             -------------------
    begin                : Jan 5, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
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

#ifndef TNLMESHFUNCTIONEVALUATOR_IMPL_H
#define	TNLMESHFUNCTIONEVALUATOR_IMPL_H

#include <functions/tnlMeshFunctionEvaluator.h>
#include <mesh/tnlTraverser.h>

template< typename OutMeshFunction,
          typename InFunction >
void
tnlMeshFunctionEvaluator< OutMeshFunction, InFunction >::
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
         return evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, all );
         break;
      case MeshInteriorDomain:
         return evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, interior );
         break;
      case MeshBoundaryDomain:
         return evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, boundary );
         break;
   }         
}


template< typename OutMeshFunction,
          typename InFunction >
void
tnlMeshFunctionEvaluator< OutMeshFunction, InFunction >::
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
tnlMeshFunctionEvaluator< OutMeshFunction, InFunction >::
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
tnlMeshFunctionEvaluator< OutMeshFunction, InFunction >::
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
tnlMeshFunctionEvaluator< OutMeshFunction, InFunction >::
evaluateEntities( OutMeshFunction& meshFunction,
                  const InFunction& function,
                  const RealType& time,
                  const RealType& outFunctionMultiplicator,
                  const RealType& inFunctionMultiplicator,
                  EntitiesType entitiesType )
{
   typedef typename MeshType::template MeshEntity< OutMeshFunction::getMeshEntityDimensions() > MeshEntityType;
   typedef tnlMeshFunctionEvaluatorAssignmentEntitiesProcessor< MeshType, TraverserUserData > AssignmentEntitiesProcessor;
   typedef tnlMeshFunctionEvaluatorAdditionEntitiesProcessor< MeshType, TraverserUserData > AdditionEntitiesProcessor;
  
   if( std::is_same< MeshDeviceType, tnlHost >::value )
   {
      TraverserUserData userData( &function, &time, &meshFunction, &outFunctionMultiplicator, &inFunctionMultiplicator );
      tnlTraverser< MeshType, MeshEntityType > meshTraverser;
      switch( entitiesType )
      {
         case all:            
            if( outFunctionMultiplicator )
               meshTraverser.template processAllEntities< TraverserUserData,
                                                          AdditionEntitiesProcessor >
                                                        ( meshFunction.getMesh(),
                                                          userData );
            else
               meshTraverser.template processAllEntities< TraverserUserData,
                                                         AssignmentEntitiesProcessor >
                                                       ( meshFunction.getMesh(),
                                                         userData );
            break;
         case interior:
            if( outFunctionMultiplicator )
               meshTraverser.template processInteriorEntities< TraverserUserData,
                                                               AdditionEntitiesProcessor >
                                                             ( meshFunction.getMesh(),
                                                               userData );
            else
               meshTraverser.template processInteriorEntities< TraverserUserData,
                                                               AssignmentEntitiesProcessor >
                                                             ( meshFunction.getMesh(),
                                                               userData );            
            break;
         case boundary:
            if( outFunctionMultiplicator )
               meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                               AdditionEntitiesProcessor >
                                                             ( meshFunction.getMesh(),
                                                               userData );
            else
               meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                               AssignmentEntitiesProcessor >
                                                             ( meshFunction.getMesh(),
                                                               userData );
            break;
      }
   }
   if( std::is_same< MeshDeviceType, tnlCuda >::value )
   {      
      OutMeshFunction* kernelMeshFunction = tnlCuda::passToDevice( meshFunction );
      InFunction* kernelFunction = tnlCuda::passToDevice( function );
      RealType* kernelTime = tnlCuda::passToDevice( time );
      RealType* kernelOutFunctionMultiplicator = tnlCuda::passToDevice( outFunctionMultiplicator );
      RealType* kernelInFunctionMultiplicator = tnlCuda::passToDevice( inFunctionMultiplicator );
      
      TraverserUserData userData( kernelFunction, kernelTime, kernelMeshFunction, kernelOutFunctionMultiplicator, kernelInFunctionMultiplicator );
      checkCudaDevice;
      tnlTraverser< MeshType, MeshEntityType > meshTraverser;
      switch( entitiesType )
      {
         case all:            
            if( outFunctionMultiplicator )
               meshTraverser.template processAllEntities< TraverserUserData,
                                                          AdditionEntitiesProcessor >
                                                        ( meshFunction.getMesh(),
                                                          userData );
            else
               meshTraverser.template processAllEntities< TraverserUserData,
                                                         AssignmentEntitiesProcessor >
                                                       ( meshFunction.getMesh(),
                                                         userData );
            break;
         case interior:
            if( outFunctionMultiplicator )
               meshTraverser.template processInteriorEntities< TraverserUserData,
                                                               AdditionEntitiesProcessor >
                                                             ( meshFunction.getMesh(),
                                                               userData );
            else
               meshTraverser.template processInteriorEntities< TraverserUserData,
                                                               AssignmentEntitiesProcessor >
                                                             ( meshFunction.getMesh(),
                                                               userData );            
            break;
         case boundary:
            if( outFunctionMultiplicator )
               meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                               AdditionEntitiesProcessor >
                                                             ( meshFunction.getMesh(),
                                                               userData );
            else
               meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                               AssignmentEntitiesProcessor >
                                                             ( meshFunction.getMesh(),
                                                               userData );
            break;         
      }      

      checkCudaDevice;      
      tnlCuda::freeFromDevice( kernelMeshFunction );
      tnlCuda::freeFromDevice( kernelFunction );
      tnlCuda::freeFromDevice( kernelTime );
      tnlCuda::freeFromDevice( kernelOutFunctionMultiplicator );
      tnlCuda::freeFromDevice( kernelInFunctionMultiplicator );            
      checkCudaDevice;
   }
}

#endif	/* TNLMESHFUNCTIONEVALUATOR_IMPL_H */

