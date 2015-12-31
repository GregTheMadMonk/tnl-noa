/***************************************************************************
                          tnlExactOperatorEvaluator.h  -  description
                             -------------------
    begin                : Nov 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLEXACTOPERATOREVALUATOR_H_
#define TNLEXACTOPERATOREVALUATOR_H_

#include <mesh/tnlGrid.h>

template< typename Real,
          typename DofVector,
          typename DifferentialOperator,
          typename Function,
          typename BoundaryConditions >
class tnlExactOperatorEvaluatorTraversalUserData
{
   public:

      const Real &time;

      const DifferentialOperator& differentialOperator;

      const BoundaryConditions& boundaryConditions;

      const Function& function;

      DofVector &fu;

      tnlExactOperatorEvaluatorTraversalUserData( const Real& time,
                                                  const DifferentialOperator& differentialOperator,
                                                  const Function& function,
                                                  const BoundaryConditions& boundaryConditions,
                                                  DofVector& fu )
      : time( time ),
        differentialOperator( differentialOperator ),
        boundaryConditions( boundaryConditions ),
        function( function ),
        fu( fu )
      {};
};

template< typename Mesh,
          typename DofVector,
          typename DifferentialOperator,
          typename Function,
          typename BoundaryConditions >
class tnlExactOperatorEvaluator
{
   public:
      typedef Mesh MeshType;
      typedef typename DofVector::RealType RealType;
      typedef typename DofVector::DeviceType DeviceType;
      typedef typename DofVector::IndexType IndexType;
      typedef tnlExactOperatorEvaluatorTraversalUserData< RealType,
                                                          DofVector,
                                                          DifferentialOperator,
                                                          Function,
                                                          BoundaryConditions > TraversalUserData;

      template< typename EntityType >
      void evaluate( const RealType& time,
                     const MeshType& mesh,
                     const DifferentialOperator& differentialOperator,
                     const Function& function,
                     const BoundaryConditions& boundaryConditions,
                     DofVector& fu ) const;

      class TraversalBoundaryEntitiesProcessor
      {
         public:

            template< int EntityDimension >
            __cuda_callable__
            static void processEntity( const MeshType& mesh,
                                       TraversalUserData& userData,
                                       const IndexType index )
            {
               userData.boundaryConditions.setBoundaryConditions( userData.time,
                                                                  mesh,
                                                                  index,
                                                                  userData.u,
                                                                  userData.fu );
            }

      };

      class TraversalInteriorEntitiesProcessor
      {
         public:

            template< int EntityDimensions >
            __cuda_callable__
            static void processEntity( const MeshType& mesh,
                                       TraversalUserData& userData,
                                       const IndexType index )
            {
               userData.fu[ index ] = 
                  userData.differentialOperator.template getValue
                  ( userData.function,
                    mesh.template getEntityCenter< EntityDimensions >( index ),
                    userData.time );
            }

      };

};

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename DofVector,
          typename DifferentialOperator,
          typename Function,
          typename BoundaryConditions >
class tnlExactOperatorEvaluator< tnlGrid< Dimensions, Real, Device, Index >, DofVector, DifferentialOperator, Function, BoundaryConditions >
{
   public:
      typedef tnlGrid< Dimensions, Real, Device, Index > MeshType;
      typedef typename DofVector::RealType RealType;
      typedef typename DofVector::DeviceType DeviceType;
      typedef typename DofVector::IndexType IndexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef tnlExactOperatorEvaluatorTraversalUserData< RealType,
                                                          DofVector,
                                                          DifferentialOperator,
                                                          Function,
                                                          BoundaryConditions > TraversalUserData;

      template< typename EntityType >
      void evaluate( const RealType& time,
                     const MeshType& mesh,
                     const DifferentialOperator& differentialOperator,
                     const Function& function,
                     const BoundaryConditions& boundaryConditions,
                     DofVector& fu ) const;

      class TraversalBoundaryEntitiesProcessor
      {
         public:

            template< typename EntityType >
            __cuda_callable__
            static void processEntity( const MeshType& mesh,
                                       TraversalUserData& userData,
                                       const EntityType& entity )
            {
               userData.fu[ entity.getIndex() ] = userData.boundaryConditions.getValue
                  ( entity,
                    userData.time,
                    userData.fu );
            }
      };

      class TraversalInteriorEntitiesProcessor
      {
         public:
            
            template< typename EntityType >
            __cuda_callable__
            static void processEntity( const MeshType& mesh,
                                       TraversalUserData& userData,
                                       const EntityType& entity )
            {
               userData.fu[ entity.getIndex() ] = 
                  userData.differentialOperator.getValue
                     ( userData.function,
                       entity.getCenter(),                                                                           
                       userData.time );
            }

            
            __cuda_callable__
            static void processCell( const MeshType& mesh,
                                     TraversalUserData& userData,
                                     const IndexType index,
                                     const CoordinatesType& c )
            {
               userData.fu[ index ] = userData.differentialOperator.getValue( userData.function,
                                                                              mesh.template getCellCenter( index ),                                                                           
                                                                              userData.time );
            }

      };

};


#include <operators/tnlExactOperatorEvaluator_impl.h>

#endif /* TNLEXACTOPERATOREVALUATOR_H_ */
