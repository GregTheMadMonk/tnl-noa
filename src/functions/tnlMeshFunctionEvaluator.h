/***************************************************************************
                          tnlMeshFunctionEvaluator.h  -  description
                             -------------------
    begin                : Jan 1, 2016
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

#ifndef TNLMESHFUNCTIONEVALUATOR_H
#define	TNLMESHFUNCTIONEVALUATOR_H

#include <mesh/tnlGrid.h>
#include <functions/tnlMeshFunction.h>
#include <functions/tnlOperatorFunction.h>

template< typename OutMeshFunction,
          typename InFunction,
          typename Real >
class tnlMeshFunctionEvaluatorTraverserUserData;

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
class tnlMeshFunctionEvaluator : public tnlDomain< OutMeshFunction::getEntitiesDimensions(), MeshDomain >
{
   public:
      typedef typename OutMeshFunction::MeshType MeshType;
      typedef typename MeshType::RealType MeshRealType;
      typedef typename MeshType::DeviceType MeshDeviceType;
      typedef typename MeshType::IndexType MeshIndexType;
      typedef typename OutMeshFunction::RealType RealType;
      typedef tnlMeshFunctionEvaluatorTraverserUserData< OutMeshFunction, InFunction, RealType > TraverserUserData;

      
      const static int meshEntityDimensions = OutMeshFunction::getEntitiesDimensions();
      
      static_assert( MeshType::meshDimensions == InFunction::getDimensions(), 
         "Input function and the mesh of the mesh function have both different number of dimensions." );
      
      static void evaluate( OutMeshFunction& meshFunction,
                            const InFunction& function,                          
                            const RealType& time = 0.0,
                            const RealType& outFunctionMultiplicator = 0.0,
                            const RealType& inFunctionMultiplicator = 1.0 );

      static void evaluateAllEntities( OutMeshFunction& meshFunction,
                                       const InFunction& function,                          
                                       const RealType& time = 0.0,
                                       const RealType& outFunctionMultiplicator = 0.0,
                                       const RealType& inFunctionMultiplicator = 1.0 );
      
      static void evaluateInteriorEntities( OutMeshFunction& meshFunction,
                                            const InFunction& function,                          
                                            const RealType& time = 0.0,
                                            const RealType& outFunctionMultiplicator = 0.0,
                                            const RealType& inFunctionMultiplicator = 1.0 );

      static void evaluateBoundaryEntities( OutMeshFunction& meshFunction,
                                            const InFunction& function,                          
                                            const RealType& time = 0.0,
                                            const RealType& outFunctionMultiplicator = 0.0,
                                            const RealType& inFunctionMultiplicator = 1.0 );

   protected:

      enum EntitiesType { all, boundary, interior };
      
      static void evaluateEntities( OutMeshFunction& meshFunction,
                                    const InFunction& function,                          
                                    const RealType& time,
                                    const RealType& outFunctionMultiplicator,
                                    const RealType& inFunctionMultiplicator,
                                    EntitiesType entitiesType );

      
}; 

template< typename OutMeshFunction,
          typename InFunction,
          typename Real >
class tnlMeshFunctionEvaluatorTraverserUserData
{
   public:

      typedef InFunction InFunctionType;

      tnlMeshFunctionEvaluatorTraverserUserData( const InFunction* function,
                                                 const Real* time,
                                                 OutMeshFunction* meshFunction,
                                                 const Real* outFunctionMultiplicator,
                                                 const Real* inFunctionMultiplicator )
      : meshFunction( meshFunction ), function( function ), time( time ), 
        outFunctionMultiplicator( outFunctionMultiplicator ),
        inFunctionMultiplicator( inFunctionMultiplicator ){}

      OutMeshFunction* meshFunction;            
      const InFunction* function;
      const Real *time, *outFunctionMultiplicator, *inFunctionMultiplicator;

};


template< typename MeshType,
          typename UserData > 
class tnlMeshFunctionEvaluatorAssignmentEntitiesProcessor
{
   public:

      template< typename EntityType >
      __cuda_callable__
      static inline void processEntity( const MeshType& mesh,
                                        UserData& userData,
                                        const EntityType& entity )
      {
         typedef tnlFunctionAdapter< MeshType, typename UserData::InFunctionType > FunctionAdapter;
         ( *userData.meshFunction )( entity ) = 
            *userData.inFunctionMultiplicator *
            FunctionAdapter::getValue( *userData.function, entity, *userData.time );
         /*cerr << "Idx = " << entity.getIndex() 
            << " Value = " << FunctionAdapter::getValue( *userData.function, entity, *userData.time ) 
            << " stored value = " << ( *userData.meshFunction )( entity )
            << " multiplicators = " << endl;*/
      }
};

template< typename MeshType,
          typename UserData > 
class tnlMeshFunctionEvaluatorAdditionEntitiesProcessor
{
   public:

      template< typename EntityType >
      __cuda_callable__
      static inline void processEntity( const MeshType& mesh,
                                        UserData& userData,
                                        const EntityType& entity )
      {
         typedef tnlFunctionAdapter< MeshType, typename UserData::InFunctionType > FunctionAdapter;
         ( *userData.meshFunction )( entity ) = 
            *userData.outFunctionMultiplicator * ( *userData.meshFunction )( entity ) +
            *userData.inFunctionMultiplicator *
            FunctionAdapter::getValue( *userData.function, entity, *userData.time );
         /*cerr << "Idx = " << entity.getIndex() 
            << " Value = " << FunctionAdapter::getValue( *userData.function, entity, *userData.time ) 
            << " stored value = " << ( *userData.meshFunction )( entity )
            << " multiplicators = " << endl;*/
      }
};


#include <functions/tnlMeshFunctionEvaluator_impl.h>

#endif	/* TNLMESHFUNCTIONEVALUATOR_H */
