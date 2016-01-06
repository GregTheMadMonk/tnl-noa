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
evaluateAllEntities( OutMeshFunction& meshFunction,
                     const InFunction& function,                          
                     const RealType& time = 0.0,
                     const RealType& outFunctionMultiplicator = 0.0,
                     const RealType& inFunctionMultiplicator = 1.0 )
{
   return evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, all );
}

template< typename OutMeshFunction,
          typename InFunction >
void
tnlMeshFunctionEvaluator< OutMeshFunction, InFunction >::
evaluateInteriorEntities( OutMeshFunction& meshFunction,
                          const InFunction& function,                          
                          const RealType& time = 0.0,
                          const RealType& outFunctionMultiplicator = 0.0,
                          const RealType& inFunctionMultiplicator = 1.0 )
{
   return evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, interior );
}

template< typename OutMeshFunction,
          typename InFunction >
void 
tnlMeshFunctionEvaluator< OutMeshFunction, InFunction >::
evaluateBoundaryEntities( OutMeshFunction& meshFunction,
                          const InFunction& function,                          
                          const RealType& time = 0.0,
                          const RealType& outFunctionMultiplicator = 0.0,
                          const RealType& inFunctionMultiplicator = 1.0 )
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
   typedef typename MeshType::template MeshEntities< meshEntityDimensions > MeshEntityType;
   
   class AssignEntitiesProcessor
   {
      template< typename EntityType >
      __cuda_callable__
      static inline void processEntity( const MeshType& mesh,
                                        TraverserUserData& userData,
                                        const EntityType& entity )
      {
         typedef tnlFunctionAdapter< MeshType, InFunction > FunctionAdapter;
         ( *userData.meshFunction )( entity ) = 
            *userData.outFunctionMultiplicator * ( *userData.meshFunction )( entity ) +
            *userData.inFunctionMultiplicator *
            FunctionAdapter::getValue( *userData.function, entity, *userData.time );
      }
   };

   if( std::is_same< MeshDeviceType, tnlHost >::value )
   {
      TraverserUserData userData( &meshFunction, &function, &time, &outFunctionMultiplicator, &inFunctionMultiplicator );
      tnlTraverser< MeshType, MeshEntityType > meshTraverser;
      switch( entitiesType )
      {
         case all:            
            meshTraverser.template processAllEntities< TraverserUserData,
                                                       AssignEntitiesProcessor >
                                                     ( meshFunction.getMesh(),
                                                       userData );
            break;
         case interior:
            meshTraverser.template processInteriroEntities< TraverserUserData,
                                                            AssignEntitiesProcessor >
                                                          ( meshFunction.getMesh(),
                                                            userData );
            break;
         case boundary:
            meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                            AssignEntitiesProcessor >
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
            meshTraverser.template processAllEntities< TraverserUserData,
                                                       AssignEntitiesProcessor >
                                                     ( meshFunction.getMesh(),
                                                       userData );
            break;
         case interior:
            meshTraverser.template processInteriorEntities< TraverserUserData,
                                                            AssignEntitiesProcessor >
                                                          ( meshFunction.getMesh(),
                                                            userData );
            break;
         case boundary:
            meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                            AssignEntitiesProcessor >
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


template< typename OutMeshFunction,
          typename Function,
          typename Operator >
void
tnlMeshFunctionEvaluator< OutMeshFunction, tnlOperatorFunction< Operator, Function> >::
evaluateEntities( OutMeshFunction& meshFunction,
                  const OperatorFunctionType& operatorFunction,
                  const RealType& time,
                  const RealType& outFunctionMultiplicator,
                  const RealType& inFunctionMultiplicator )
{
   typedef typename MeshType::template MeshEntities< meshEntityDimensions > MeshEntityType;
   
   class AssignEntitiesProcessor
   {
      template< typename EntityType >
      __cuda_callable__
      static inline void processEntity( const MeshType& mesh,
                                        TraverserUserData& userData,
                                        const EntityType& entity )
      {
         typedef tnlFunctionAdapter< MeshType, InFunction > FunctionAdapter;
         ( *userData.meshFunction )( entity ) = 
            *userData.outFunctionMultiplicator * ( *userData.meshFunction )( entity ) +
            *userData.inFunctionMultiplicator *
            FunctionAdapter::getValue( *userData.function, entity, *userData.time );
      }
   };

   if( std::is_same< MeshDeviceType, tnlHost >::value )
   {
      TraverserUserData userData( &function, &time, &meshFunction, &outFunctionMultiplicator, &inFunctionMultiplicator );
      tnlTraverser< MeshType, MeshEntityType > meshTraverser;
      meshTraverser.template processInterirorEntities< TraverserUserData, AssignEntitiesProcessor >
         ( meshFunction.getMesh(),
           userData );
      
   }
   if( std::is_same< MeshDeviceType, tnlCuda >::value )
   {      
      OutMeshFunction* kernelMeshFunction = tnlCuda::passToDevice( meshFunction );
      Function* kernelFunction = tnlCuda::passToDevice( *operatorFunction.function );
      Operator* kernelOperator = tnlCuda::passToDevice( *operatorFunction.operator_ );
      OperatorFunctionType auxOperatorFunction( *kernelOperator, *kernelFunction );
      OperatorFunctionType* kernelOperatorFunction = tnlCuda::passToDevice( auxOperatorFunction );
      RealType* kernelTime = tnlCuda::passToDevice( time );
      RealType* kernelOutFunctionMultiplicator = tnlCuda::passToDevice( outFunctionMultiplicator );
      RealType* kernelInFunctionMultiplicator = tnlCuda::passToDevice( inFunctionMultiplicator );
      
      TraverserUserData userData( kernelOperatorFunction, kernelTime, kernelMeshFunction, kernelOutFunctionMultiplicator, kernelInFunctionMultiplicator );
      checkCudaDevice;
      tnlTraverser< MeshType, MeshEntityType > meshTraverser;
      meshTraverser.template processInteriorEntities< TraverserUserData, AssignEntitiesProcessor >
         ( meshFunction.getMesh(),
           userData );


      checkCudaDevice;      
      tnlCuda::freeFromDevice( kernelMeshFunction );
      tnlCuda::freeFromDevice( kernelFunction );
      tnlCuda::freeFromDevice( kernelOperator );
      tnlCuda::freeFromDevice( kernelOperatorFunction );
      tnlCuda::freeFromDevice( kernelTime );
      tnlCuda::freeFromDevice( kernelOutFunctionMultiplicator );
      tnlCuda::freeFromDevice( kernelInFunctionMultiplicator );
            
      checkCudaDevice;
   }
}


#endif	/* TNLMESHFUNCTIONEVALUATOR_IMPL_H */

