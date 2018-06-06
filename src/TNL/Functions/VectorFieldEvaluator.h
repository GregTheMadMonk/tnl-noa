/***************************************************************************
                          VectorFieldEvaluator.h  -  description
                             -------------------
    begin                : Dec 24, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Functions/MeshFunction.h>
#include <TNL/Functions/OperatorFunction.h>
#include <TNL/Functions/FunctionAdapter.h>

namespace TNL {
namespace Functions {   

template< typename OutVectorField,
          typename InVectorField,
          typename Real >
class VectorFieldEvaluatorTraverserUserData
{
   public:
      typedef InVectorField InVectorFieldType;

      VectorFieldEvaluatorTraverserUserData( const InVectorField* inVectorField,
                                             const Real& time,
                                             OutVectorField* outVectorField,
                                             const Real& outVectorFieldMultiplicator,
                                             const Real& inVectorFieldMultiplicator )
      : outVectorField( outVectorField ),
        inVectorField( inVectorField ),
        time( time ),
        outVectorFieldMultiplicator( outVectorFieldMultiplicator ),
        inVectorFieldMultiplicator( inVectorFieldMultiplicator )
      {}

      OutVectorField* outVectorField;
      const InVectorField* inVectorField;
      const Real time, outVectorFieldMultiplicator, inVectorFieldMultiplicator;

};


/***
 * General vector field evaluator. As an input function any type implementing
 * getValue( meshEntity, time ) may be substituted.
 * Methods:
 *  evaluate() -  evaluate the input function on its domain
 *  evaluateAllEntities() - evaluate the input function on ALL mesh entities
 *  evaluateInteriorEntities() - evaluate the input function only on the INTERIOR mesh entities
 *  evaluateBoundaryEntities() - evaluate the input function only on the BOUNDARY mesh entities
 */
template< typename OutVectorField,
          typename InVectorField >
class VectorFieldEvaluator
{
   static_assert( OutVectorField::getDomainDimension() == InVectorField::getDomainDimension(),
                  "Input and output vector field must have the same domain dimensions." );

   public:
      typedef typename OutVectorField::RealType RealType;
      typedef typename OutVectorField::MeshType MeshType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef Functions::VectorFieldEvaluatorTraverserUserData< OutVectorField, InVectorField, RealType > TraverserUserData;

      template< typename OutVectorFieldPointer, typename InVectorFieldPointer >
      static void evaluate( OutVectorFieldPointer& meshFunction,
                            const InVectorFieldPointer& function,
                            const RealType& time = 0.0,
                            const RealType& outFunctionMultiplicator = 0.0,
                            const RealType& inFunctionMultiplicator = 1.0 );

      template< typename OutVectorFieldPointer, typename InVectorFieldPointer >
      static void evaluateAllEntities( OutVectorFieldPointer& meshFunction,
                                       const InVectorFieldPointer& function,
                                       const RealType& time = 0.0,
                                       const RealType& outFunctionMultiplicator = 0.0,
                                       const RealType& inFunctionMultiplicator = 1.0 );
 
      template< typename OutVectorFieldPointer, typename InVectorFieldPointer >
      static void evaluateInteriorEntities( OutVectorFieldPointer& meshFunction,
                                            const InVectorFieldPointer& function,
                                            const RealType& time = 0.0,
                                            const RealType& outFunctionMultiplicator = 0.0,
                                            const RealType& inFunctionMultiplicator = 1.0 );

      template< typename OutVectorFieldPointer, typename InVectorFieldPointer >
      static void evaluateBoundaryEntities( OutVectorFieldPointer& meshFunction,
                                            const InVectorFieldPointer& function,
                                            const RealType& time = 0.0,
                                            const RealType& outFunctionMultiplicator = 0.0,
                                            const RealType& inFunctionMultiplicator = 1.0 );

   protected:

      enum EntitiesType { all, boundary, interior };
 
      template< typename OutVectorFieldPointer, typename InVectorFieldPointer >
      static void evaluateEntities( OutVectorFieldPointer& meshFunction,
                                    const InVectorFieldPointer& function,
                                    const RealType& time,
                                    const RealType& outFunctionMultiplicator,
                                    const RealType& inFunctionMultiplicator,
                                    EntitiesType entitiesType );

 
};


template< typename MeshType,
          typename UserData >
class VectorFieldEvaluatorAssignmentEntitiesProcessor
{
   public:

      template< typename EntityType >
      __cuda_callable__
      static inline void processEntity( const MeshType& mesh,
                                        UserData& userData,
                                        const EntityType& entity )
      {
         userData.outVectorField->setElement(
            entity.getIndex(),
            userData.inVectorFieldMultiplicator * ( *userData.inVectorField )( entity, userData.time ) );         
         
         /*typedef FunctionAdapter< MeshType, typename UserData::InVectorFieldType > FunctionAdapter;
         ( *userData.meshFunction )( entity ) =
            userData.inFunctionMultiplicator *
            FunctionAdapter::getValue( *userData.function, entity, userData.time );*/
         /*cerr << "Idx = " << entity.getIndex()
            << " Value = " << FunctionAdapter::getValue( *userData.function, entity, userData.time )
            << " stored value = " << ( *userData.meshFunction )( entity )
            << " multiplicators = " << std::endl;*/
      }
};

template< typename MeshType,
          typename UserData >
class VectorFieldEvaluatorAdditionEntitiesProcessor
{
   public:

      template< typename EntityType >
      __cuda_callable__
      static inline void processEntity( const MeshType& mesh,
                                        UserData& userData,
                                        const EntityType& entity )
      {
         const auto& i = entity.getIndex();
         const auto v = userData.outVectorFieldMultiplicator * userData.outVectorField->getElement( i );
         userData.outVectorField->setElement(
            i,
            v + userData.inVectorFieldMultiplicator * ( *userData.inVectorField )( entity, userData.time ) );
         
         /*typedef FunctionAdapter< MeshType, typename UserData::InVectorFieldType > FunctionAdapter;
         ( *userData.meshFunction )( entity ) =
            userData.outFunctionMultiplicator * ( *userData.meshFunction )( entity ) +
            userData.inFunctionMultiplicator *
            FunctionAdapter::getValue( *userData.function, entity, userData.time );*/
         /*cerr << "Idx = " << entity.getIndex()
            << " Value = " << FunctionAdapter::getValue( *userData.function, entity, userData.time )
            << " stored value = " << ( *userData.meshFunction )( entity )
            << " multiplicators = " << std::endl;*/
      }
};

} // namespace Functions
} // namespace TNL

#include <TNL/Functions/VectorFieldEvaluator_impl.h>
