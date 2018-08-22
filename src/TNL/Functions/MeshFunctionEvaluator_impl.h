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
#include <TNL/Meshes/Traverser.h>

namespace TNL {
namespace Functions {   

template< typename OutMeshFunction,
          typename InFunction >
   template< typename OutMeshFunctionPointer, typename InFunctionPointer >
void
MeshFunctionEvaluator< OutMeshFunction, InFunction >::
evaluate( OutMeshFunctionPointer& meshFunction,
          const InFunctionPointer& function,
          const RealType& time,
          const RealType& outFunctionMultiplicator,
          const RealType& inFunctionMultiplicator )
{
   static_assert( std::is_same< typename std::decay< typename OutMeshFunctionPointer::ObjectType >::type, OutMeshFunction >::value, "expected a smart pointer" );
   static_assert( std::is_same< typename std::decay< typename InFunctionPointer::ObjectType >::type, InFunction >::value, "expected a smart pointer" );

   switch( InFunction::getDomainType() )
   {
      case NonspaceDomain:
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
   template< typename OutMeshFunctionPointer, typename InFunctionPointer >
void
MeshFunctionEvaluator< OutMeshFunction, InFunction >::
evaluateAllEntities( OutMeshFunctionPointer& meshFunction,
                     const InFunctionPointer& function,
                     const RealType& time,
                     const RealType& outFunctionMultiplicator,
                     const RealType& inFunctionMultiplicator )
{
   static_assert( std::is_same< typename std::decay< typename OutMeshFunctionPointer::ObjectType >::type, OutMeshFunction >::value, "expected a smart pointer" );
   static_assert( std::is_same< typename std::decay< typename InFunctionPointer::ObjectType >::type, InFunction >::value, "expected a smart pointer" );

   return evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, all );
}

template< typename OutMeshFunction,
          typename InFunction >
   template< typename OutMeshFunctionPointer, typename InFunctionPointer >
void
MeshFunctionEvaluator< OutMeshFunction, InFunction >::
evaluateInteriorEntities( OutMeshFunctionPointer& meshFunction,
                          const InFunctionPointer& function,
                          const RealType& time,
                          const RealType& outFunctionMultiplicator,
                          const RealType& inFunctionMultiplicator )
{
   static_assert( std::is_same< typename std::decay< typename OutMeshFunctionPointer::ObjectType >::type, OutMeshFunction >::value, "expected a smart pointer" );
   static_assert( std::is_same< typename std::decay< typename InFunctionPointer::ObjectType >::type, InFunction >::value, "expected a smart pointer" );

   return evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, interior );
}

template< typename OutMeshFunction,
          typename InFunction >
   template< typename OutMeshFunctionPointer, typename InFunctionPointer >
void
MeshFunctionEvaluator< OutMeshFunction, InFunction >::
evaluateBoundaryEntities( OutMeshFunctionPointer& meshFunction,
                          const InFunctionPointer& function,
                          const RealType& time,
                          const RealType& outFunctionMultiplicator,
                          const RealType& inFunctionMultiplicator )
{
   static_assert( std::is_same< typename std::decay< typename OutMeshFunctionPointer::ObjectType >::type, OutMeshFunction >::value, "expected a smart pointer" );
   static_assert( std::is_same< typename std::decay< typename InFunctionPointer::ObjectType >::type, InFunction >::value, "expected a smart pointer" );

   return evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, boundary );
}



template< typename OutMeshFunction,
          typename InFunction >
   template< typename OutMeshFunctionPointer, typename InFunctionPointer >
void
MeshFunctionEvaluator< OutMeshFunction, InFunction >::
evaluateEntities( OutMeshFunctionPointer& meshFunction,
                  const InFunctionPointer& function,
                  const RealType& time,
                  const RealType& outFunctionMultiplicator,
                  const RealType& inFunctionMultiplicator,
                  EntitiesType entitiesType )
{
   static_assert( std::is_same< typename std::decay< typename OutMeshFunctionPointer::ObjectType >::type, OutMeshFunction >::value, "expected a smart pointer" );
   static_assert( std::is_same< typename std::decay< typename InFunctionPointer::ObjectType >::type, InFunction >::value, "expected a smart pointer" );

   typedef typename MeshType::template EntityType< OutMeshFunction::getEntitiesDimension() > MeshEntityType;
   typedef Functions::MeshFunctionEvaluatorAssignmentEntitiesProcessor< MeshType, TraverserUserData > AssignmentEntitiesProcessor;
   typedef Functions::MeshFunctionEvaluatorAdditionEntitiesProcessor< MeshType, TraverserUserData > AdditionEntitiesProcessor;
   //typedef typename OutMeshFunction::MeshPointer OutMeshPointer;
   typedef Pointers::SharedPointer<  TraverserUserData, DeviceType > TraverserUserDataPointer;
   
   Pointers::SharedPointer<  TraverserUserData, DeviceType >
      userData( &function.template getData< DeviceType >(),
                time,
                &meshFunction.template modifyData< DeviceType >(),
                outFunctionMultiplicator,
                inFunctionMultiplicator );
   Meshes::Traverser< MeshType, MeshEntityType > meshTraverser;
   switch( entitiesType )
   {
      case all:
         if( outFunctionMultiplicator )
            meshTraverser.template processAllEntities< TraverserUserData,
                                                       AdditionEntitiesProcessor >
                                                     ( meshFunction->getMeshPointer(),
                                                       userData );
         else
            meshTraverser.template processAllEntities< TraverserUserData,
                                                       AssignmentEntitiesProcessor >
                                                    ( meshFunction->getMeshPointer(),
                                                      userData );
         break;
      case interior:
         if( outFunctionMultiplicator )
            meshTraverser.template processInteriorEntities< TraverserUserData,
                                                            AdditionEntitiesProcessor >
                                                          ( meshFunction->getMeshPointer(),
                                                            userData );
         else
            meshTraverser.template processInteriorEntities< TraverserUserData,
                                                            AssignmentEntitiesProcessor >
                                                          ( meshFunction->getMeshPointer(),
                                                            userData );
         break;
      case boundary:
         if( outFunctionMultiplicator )
            meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                            AdditionEntitiesProcessor >
                                                          ( meshFunction->getMeshPointer(),
                                                            userData );
         else
            meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                            AssignmentEntitiesProcessor >
                                                          ( meshFunction->getMeshPointer(),
                                                            userData );
         break;
   }
}

} // namespace Functions
} // namespace TNL
