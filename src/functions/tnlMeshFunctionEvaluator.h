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
          typename InFunction >
class tnlMeshFunctionEvaluator
{
   public:
      typedef typename OutMeshFunction::MeshType MeshType;
      typedef typename MeshType::RealType MeshRealType;
      typedef typename MeshType::DeviceType MeshDeviceType;
      typedef typename MeshType::IndexType MeshIndexType;
      typedef typename OutMeshFunction::Real RealType;
      
      const static int meshEntityDimensions = OutMeshFunction::entityDimensions;
      
      static_assert( MeshType::meshDimensions == InFunction::Dimensions, 
         "Input function and the mesh of the mesh function have both different number of dimensions." );
      
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

      
      class TraverserUserData
      {
         public:
            TraverserUserData( const InFunction* function,
                               const RealType* time,
                               OutMeshFunction* meshFunction,
                               const RealType* outFunctionMultiplicator,
                               const RealType* inFunctionMultiplicator )
            : meshFunction( meshFunction ), function( function ), time( time ), 
              outFunctionMultiplicator( outFunctionMultiplicator ),
              inFunctionMultiplicator( inFunctionMultiplicator ){}

         protected:
            OutMeshFunction* meshFunction;            
            const InFunction* function;
            const RealType *time, *outFunctionMultiplicator, *inFunctionMultiplicator;
            
      };
}; 

template< typename OutMeshFunction,
          typename Operator,
          typename Function >
class tnlMeshFunctionEvaluator< OutMeshFunction, tnlOperatorFunction< Operator, Function > >
{
   public:
      
      typedef typename OutMeshFunction::MeshType MeshType;
      typedef typename MeshType::RealType MeshRealType;
      typedef typename MeshType::DeviceType MeshDeviceType;
      typedef typename MeshType::IndexType MeshIndexType;
      typedef typename OutMeshFunction::Real RealType;
      typedef tnlOperatorFunction< Operator, Function > OperatorFunctionType;
      
      static void evaluateAllEntities( OutMeshFunction& meshFunction,
                                       const OperatorFunctionType& function,                          
                                       const RealType& time = 0.0,
                                       const RealType& outFunctionMultiplicator = 0.0,
                                       const RealType& inFunctionMultiplicator = 1.0 )
      {
         evaluateInteriorEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator );
      };
      
      static void evaluateInteriorEntities( OutMeshFunction& meshFunction,
                                            const OperatorFunctionType& function,                          
                                            const RealType& time = 0.0,
                                            const RealType& outFunctionMultiplicator = 0.0,
                                            const RealType& inFunctionMultiplicator = 1.0 );
      
      class TraverserUserData
      {
         public:
            TraverserUserData( const OperatorFunctionType* operatorFunction,              
                               const RealType* time,
                               OutMeshFunction* meshFunction,
                               const RealType* outFunctionMultiplicator,
                               const RealType* inFunctionMultiplicator )
            : meshFunction( meshFunction ), operatorFunction( operatorFunction ),time( time ), 
              outFunctionMultiplicator( outFunctionMultiplicator ),
              inFunctionMultiplicator( inFunctionMultiplicator ){}

         protected:
            OutMeshFunction* meshFunction;            
            const OperatorFunctionType* operatorFunction;
            const RealType *time, *outFunctionMultiplicator, *inFunctionMultiplicator;
            
      };

};
#endif	/* TNLMESHFUNCTIONEVALUATOR_H */

