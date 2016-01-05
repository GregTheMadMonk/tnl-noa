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

template< typename OutMeshFunction,
          typename InFunction >
void
tnlMeshFunctionEvaluator< OutMeshFunction, InFunction >::
assign( OutMeshFunction& meshFunction,
        const InFunction& function,
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
         ( * userData.meshFunction )( entity ) = 
               FunctionAdapter::getValue(
                  *userData.function,
                  entity,
                  *userData.time );
      }
   };

   if( std::is_same< MeshDeviceType, tnlHost >::value )
   {
      TraverserUserData userData( &meshFunction, &function, &time, &outFunctionMultiplicator, &inFunctionMultiplicator );
      tnlTraverser< MeshType, MeshEntityType > meshTraverser;
      meshTraverser.template processAllEntities< TraverserUserData,
                                                 AssignEntitiesProcessor >
                                                ( meshFunction.getMesh(),
                                                  userData );
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
      meshTraverser.template processAllEntities< TraverserUserData,
                                                 AssignEntitiesProcessor >
                                                ( meshFunction.getMesh(),
                                                  userData );
      

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

