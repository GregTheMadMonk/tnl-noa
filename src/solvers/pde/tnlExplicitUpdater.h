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
          typename BoundaryConditions,
          typename InteriorUpdater >
class tnlExplicitUpdaterTraversalUserData
{
   public:
      const Real &time, &tau;

      BoundaryConditions& boundaryConditions;

      InteriorUpdater& interiorUpdater;

      DofVector &u, &fu;

      tnlExplicitUpdaterTraversalUserData( const Real& time,
                                           const Real& tau,
                                           BoundaryConditions& boundaryConditions,
                                           InteriorUpdater& interiorUpdater,
                                           DofVector& u,
                                           DofVector& fu )
      : time( time ),
        tau( tau ),
        boundaryConditions( boundaryConditions ),
        interiorUpdater( interiorUpdater ),
        u( u ),
        fu( fu )
      {};
};

template< typename Mesh,
          typename DofVector,
          typename BoundaryConditions,
          typename InteriorUpdater >
class tnlExplicitUpdater
{
   public:
      typedef Mesh MeshType;
      typedef typename DofVector::RealType RealType;
      typedef typename DofVector::DeviceType DeviceType;
      typedef typename DofVector::IndexType IndexType;
      typedef tnlExplicitUpdaterTraversalUserData< RealType,
                                                   DofVector,
                                                   BoundaryConditions,
                                                   InteriorUpdater > TraversalUserData;

      template< int EntityDimensions >
      void update( const RealType& time,
                   const RealType& tau,
                   const MeshType& mesh,
                   BoundaryConditions& boundaryConditions,
                   InteriorUpdater& interiorUpdater,
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
                                                                  userData.tau,
                                                                  mesh,
                                                                  index,
                                                                  userData.u,
                                                                  userData.fu );
            }

      };

      class TraversalInteriorEntitiesProcessor
      {
         public:

            template< int EntityDimension >
            void processEntity( const MeshType& mesh,
                                TraversalUserData& userData,
                                const IndexType index )
            {
               userData.boundaryConditions.update( userData.time,
                                                   userData.tau,
                                                   mesh,
                                                   index,
                                                   userData.u,
                                                   userData.fu );
            }

      };

};

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename DofVector,
          typename BoundaryConditions,
          typename InteriorUpdater >
class tnlExplicitUpdater< tnlGrid< Dimensions, Real, Device, Index >,
                          DofVector,
                          BoundaryConditions,
                          InteriorUpdater >
{
   public:

      typedef tnlGrid< Dimensions, Real, Device, Index > MeshType;
      typedef typename MeshType::RealType RealType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef tnlExplicitUpdaterTraversalUserData< RealType,
                                                   DofVector,
                                                   BoundaryConditions,
                                                   InteriorUpdater > TraversalUserData;
      
      template< int EntityDimensions >
      void update( const RealType& time,
                   const RealType& tau,
                   const MeshType& mesh,
                   BoundaryConditions& boundaryConditions,
                   InteriorUpdater& interiorUpdater,
                   DofVector& u,
                   DofVector& fu ) const;

      class TraversalBoundaryEntitiesProcessor
      {
         public:

            template< int EntityDimension >
            void processEntity( const MeshType& mesh,
                                TraversalUserData& userData,
                                const IndexType index,
                                const CoordinatesType& coordinates )
            {
               userData.boundaryConditions.setBoundaryConditions( userData.time,
                                                                  userData.tau,
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

            template< int EntityDimension >
            void processEntity( const MeshType& mesh,
                                TraversalUserData& userData,
                                const IndexType index,
                                const CoordinatesType& coordinates )
            {
               userData.interiorUpdater.update( userData.time,
                                                userData.tau,
                                                mesh,
                                                index,
                                                coordinates,
                                                userData.u,
                                                userData.fu );
            }

      };

};

#include <implementation/solvers/pde/tnlExplicitUpdater_impl.h>
#endif /* TNLEXPLICITUPDATER_H_ */
