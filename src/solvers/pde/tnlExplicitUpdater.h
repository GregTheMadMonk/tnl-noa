/***************************************************************************
                          tnlExplicitUpdater.h  -  description
                             -------------------
    begin                : Jul 29, 2014
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

#ifndef TNLEXPLICITUPDATER_H_
#define TNLEXPLICITUPDATER_H_

template< typename Real,
          typename DofVector,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide >
class tnlExplicitUpdaterTraversalUserData
{
   public:

      const Real &time;

      DifferentialOperator& differentialOperator;

      BoundaryConditions& boundaryConditions;

      RightHandSide& rightHandSide;

      DofVector &u, &fu;

      tnlExplicitUpdaterTraversalUserData( const Real& time,
                                           DifferentialOperator& differentialOperator,
                                           BoundaryConditions& boundaryConditions,
                                           RightHandSide& rightHandSide,
                                           DofVector& u,
                                           DofVector& fu )
      : time( time ),
        differentialOperator( differentialOperator ),
        boundaryConditions( boundaryConditions ),
        rightHandSide( rightHandSide ),
        u( u ),
        fu( fu )
      {};

   protected:



};

template< typename Mesh,
          typename DofVector,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide >
class tnlExplicitUpdater
{
   public:
      typedef Mesh MeshType;
      typedef typename DofVector::RealType RealType;
      typedef typename DofVector::DeviceType DeviceType;
      typedef typename DofVector::IndexType IndexType;
      typedef tnlExplicitUpdaterTraversalUserData< RealType,
                                                   DofVector,
                                                   DifferentialOperator,
                                                   BoundaryConditions,
                                                   RightHandSide > TraversalUserData;

      template< int EntityDimensions >
      void update( const RealType& time,
                   const MeshType& mesh,
                   DifferentialOperator& differentialOperator,
                   BoundaryConditions& boundaryConditions,
                   RightHandSide& rightHandSide,
                   DofVector& u,
                   DofVector& fu ) const;

      class TraversalBoundaryEntitiesProcessor
      {
         public:

            template< int EntityDimension >
            void processEntity( const MeshType& mesh,
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
            void processEntity( const MeshType& mesh,
                                TraversalUserData& userData,
                                const IndexType index )
            {
               userData.fu[ index ] = userData.differentialOperator.getValue( mesh,
                                                                               index,
                                                                               userData.u,
                                                                               userData.time );
               userData.fu[ index ] += userData.rightHandSide.getValue( mesh.getEntityCenter< EntityDimensions >( index ),
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
          typename BoundaryConditions,
          typename RightHandSide >
class tnlExplicitUpdater< tnlGrid< Dimensions, Real, Device, Index >,
                          DofVector,
                          DifferentialOperator,
                          BoundaryConditions,
                          RightHandSide >
{
   public:

      typedef tnlGrid< Dimensions, Real, Device, Index > MeshType;
      typedef typename MeshType::RealType RealType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef tnlExplicitUpdaterTraversalUserData< RealType,
                                                   DofVector,
                                                   DifferentialOperator,
                                                   BoundaryConditions,
                                                   RightHandSide > TraversalUserData;
      
      template< int EntityDimensions >
      void update( const RealType& time,
                   const MeshType& mesh,
                   DifferentialOperator& differentialOperator,
                   BoundaryConditions& boundaryConditions,
                   RightHandSide& rightHandSide,
                   DofVector& u,
                   DofVector& fu ) const;

      class TraversalBoundaryEntitiesProcessor
      {
         public:

            /****
             * TODO: This must be specialized for entities with different dimensions
             * otherwise 'coordinates' would not make sense without knowing the orientation.
             */
            template< int EntityDimension >
            void processEntity( const MeshType& mesh,
                                TraversalUserData& userData,
                                const IndexType index,
                                const CoordinatesType& coordinates )
            {
               userData.boundaryConditions.setBoundaryConditions( userData.time,
                                                                  mesh,
                                                                  index,
                                                                  coordinates,
                                                                  userData.u,
                                                                  userData.fu );
            }

      };

      class TraversalInteriorEntitiesProcessor
      {
         public:

            template< int EntityDimensions >
            void processEntity( const MeshType& mesh,
                                TraversalUserData& userData,
                                const IndexType index,
                                const CoordinatesType& coordinates )
            {
               userData.fu[ index ] = userData.differentialOperator.getValue( mesh,
                                                                               index,
                                                                               coordinates,
                                                                               userData.u,
                                                                               userData.time );

               userData.fu[ index ] += userData.rightHandSide.getValue( mesh.getCellCenter( coordinates ),
                                                                        userData.time );
            }

      };

};

#include <implementation/solvers/pde/tnlExplicitUpdater_impl.h>
#endif /* TNLEXPLICITUPDATER_H_ */
