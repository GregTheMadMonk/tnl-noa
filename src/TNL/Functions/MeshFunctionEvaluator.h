/***************************************************************************
                          MeshFunctionEvaluator.h  -  description
                             -------------------
    begin                : Jan 1, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Functions/MeshFunction.h>
#include <TNL/Functions/OperatorFunction.h>
#include <TNL/Functions/FunctionAdapter.h>

namespace TNL {
namespace Functions {   

template< typename OutMeshFunction,
          typename InFunction,
          typename Real >
class MeshFunctionEvaluatorTraverserUserData
{
   public:
      typedef InFunction InFunctionType;

      MeshFunctionEvaluatorTraverserUserData( const InFunction* function,
                                              const Real& time,
                                              OutMeshFunction* meshFunction,
                                              const Real& outFunctionMultiplicator,
                                              const Real& inFunctionMultiplicator )
      : meshFunction( meshFunction ),
        function( function ),
        time( time ),
        outFunctionMultiplicator( outFunctionMultiplicator ),
        inFunctionMultiplicator( inFunctionMultiplicator )
      {}

      OutMeshFunction* meshFunction;
      const InFunction* function;
      const Real time, outFunctionMultiplicator, inFunctionMultiplicator;

};


/***
 * General mesh function evaluator. As an input function any type implementing
 * getValue( meshEntity, time ) may be substituted.
 * Methods:
 *  evaluate() -  evaluate the input function on its domain
 *  evaluateAllEntities() - evaluate the input function on ALL mesh entities
 *  evaluateInteriorEntities() - evaluate the input function only on the INTERIOR mesh entities
 *  evaluateBoundaryEntities() - evaluate the input function only on the BOUNDARY mesh entities
 */
template< typename OutMeshFunction,
          typename InFunction >
class MeshFunctionEvaluator
{
   static_assert( OutMeshFunction::getDomainDimension() == InFunction::getDomainDimension(),
                  "Input and output functions must have the same domain dimensions." );

   public:
      typedef typename OutMeshFunction::RealType RealType;
      typedef typename OutMeshFunction::MeshType MeshType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef Functions::MeshFunctionEvaluatorTraverserUserData< OutMeshFunction, InFunction, RealType > TraverserUserData;

      template< typename OutMeshFunctionPointer, typename InFunctionPointer >
      static void evaluate( OutMeshFunctionPointer& meshFunction,
                            const InFunctionPointer& function,
                            const RealType& time = 0.0,
                            const RealType& outFunctionMultiplicator = 0.0,
                            const RealType& inFunctionMultiplicator = 1.0 );

      template< typename OutMeshFunctionPointer, typename InFunctionPointer >
      static void evaluateAllEntities( OutMeshFunctionPointer& meshFunction,
                                       const InFunctionPointer& function,
                                       const RealType& time = 0.0,
                                       const RealType& outFunctionMultiplicator = 0.0,
                                       const RealType& inFunctionMultiplicator = 1.0 );
 
      template< typename OutMeshFunctionPointer, typename InFunctionPointer >
      static void evaluateInteriorEntities( OutMeshFunctionPointer& meshFunction,
                                            const InFunctionPointer& function,
                                            const RealType& time = 0.0,
                                            const RealType& outFunctionMultiplicator = 0.0,
                                            const RealType& inFunctionMultiplicator = 1.0 );

      template< typename OutMeshFunctionPointer, typename InFunctionPointer >
      static void evaluateBoundaryEntities( OutMeshFunctionPointer& meshFunction,
                                            const InFunctionPointer& function,
                                            const RealType& time = 0.0,
                                            const RealType& outFunctionMultiplicator = 0.0,
                                            const RealType& inFunctionMultiplicator = 1.0 );

   protected:

      enum EntitiesType { all, boundary, interior };
 
      template< typename OutMeshFunctionPointer, typename InFunctionPointer >
      static void evaluateEntities( OutMeshFunctionPointer& meshFunction,
                                    const InFunctionPointer& function,
                                    const RealType& time,
                                    const RealType& outFunctionMultiplicator,
                                    const RealType& inFunctionMultiplicator,
                                    EntitiesType entitiesType );

 
};


template< typename MeshType,
          typename UserData >
class MeshFunctionEvaluatorAssignmentEntitiesProcessor
{
   public:

      template< typename EntityType >
      __cuda_callable__
      static inline void processEntity( const MeshType& mesh,
                                        UserData& userData,
                                        const EntityType& entity )
      {
         typedef FunctionAdapter< MeshType, typename UserData::InFunctionType > FunctionAdapter;
         ( *userData.meshFunction )( entity ) =
            userData.inFunctionMultiplicator *
            FunctionAdapter::getValue( *userData.function, entity, userData.time );
         /*cerr << "Idx = " << entity.getIndex()
            << " Value = " << FunctionAdapter::getValue( *userData.function, entity, userData.time )
            << " stored value = " << ( *userData.meshFunction )( entity )
            << " multiplicators = " << std::endl;*/
      }
};

template< typename MeshType,
          typename UserData >
class MeshFunctionEvaluatorAdditionEntitiesProcessor
{
   public:

      template< typename EntityType >
      __cuda_callable__
      static inline void processEntity( const MeshType& mesh,
                                        UserData& userData,
                                        const EntityType& entity )
      {
         typedef FunctionAdapter< MeshType, typename UserData::InFunctionType > FunctionAdapter;
         ( *userData.meshFunction )( entity ) =
            userData.outFunctionMultiplicator * ( *userData.meshFunction )( entity ) +
            userData.inFunctionMultiplicator *
            FunctionAdapter::getValue( *userData.function, entity, userData.time );
         /*cerr << "Idx = " << entity.getIndex()
            << " Value = " << FunctionAdapter::getValue( *userData.function, entity, userData.time )
            << " stored value = " << ( *userData.meshFunction )( entity )
            << " multiplicators = " << std::endl;*/
      }
};

} // namespace Functions
} // namespace TNL

#include <TNL/Functions/MeshFunctionEvaluator_impl.h>
