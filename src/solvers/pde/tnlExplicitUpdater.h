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

#include <functors/tnlFunctionAdapter.h>

template< typename Real,
          typename DofVector,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide >
class tnlExplicitUpdaterTraverserUserData
{
   public:

      const Real *time;

      const DifferentialOperator* differentialOperator;

      const BoundaryConditions* boundaryConditions;

      const RightHandSide* rightHandSide;

      DofVector *u, *fu;

      tnlExplicitUpdaterTraverserUserData( const Real& time,
                                           const DifferentialOperator& differentialOperator,
                                           const BoundaryConditions& boundaryConditions,
                                           const RightHandSide& rightHandSide,
                                           DofVector& u,
                                           DofVector& fu )
      : time( &time ),
        differentialOperator( &differentialOperator ),
        boundaryConditions( &boundaryConditions ),
        rightHandSide( &rightHandSide ),
        u( &u ),
        fu( &fu )
      {};
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
      typedef tnlExplicitUpdaterTraverserUserData< RealType,
                                                   DofVector,
                                                   DifferentialOperator,
                                                   BoundaryConditions,
                                                   RightHandSide > TraverserUserData;

      template< int EntityDimensions >
      void update( const RealType& time,
                   const MeshType& mesh,
                   DifferentialOperator& differentialOperator,
                   BoundaryConditions& boundaryConditions,
                   RightHandSide& rightHandSide,
                   DofVector& u,
                   DofVector& fu ) const;

      class TraverserBoundaryEntitiesProcessor
      {
         public:

            template< int EntityDimension >
#ifdef HAVE_CUDA
            __host__ __device__
#endif
            static void processEntity( const MeshType& mesh,
                                       TraverserUserData& userData,
                                       const IndexType index )
            {
               userData.boundaryConditions->setBoundaryConditions( *userData.time,
                                                                   mesh,
                                                                   index,
                                                                   *userData.u,
                                                                   *userData.fu );
            }

      };

      class TraverserInteriorEntitiesProcessor
      {
         public:

            template< int EntityDimensions >
#ifdef HAVE_CUDA
            __host__ __device__
#endif
            static void processEntity( const MeshType& mesh,
                                       TraverserUserData& userData,
                                       const IndexType index )
            {
               (* userData.fu )[ index ] = userData.differentialOperator->getValue( mesh,
                                                                                    index,
                                                                                    *userData.u,
                                                                                    *userData.time );
               typedef tnlFunctionAdapter< MeshType, RightHandSide > FunctionAdapter;
               ( *userData.fu )[ index ] += FunctionAdapter::getValue( mesh,
                                                                       *userData.rightHandSide,
                                                                       index,
                                                                       *userData.time );
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
      typedef tnlExplicitUpdaterTraverserUserData< RealType,
                                                   DofVector,
                                                   DifferentialOperator,
                                                   BoundaryConditions,
                                                   RightHandSide > TraverserUserData;
      
      template< int EntityDimensions >
      void update( const RealType& time,
                   const MeshType& mesh,
                   const DifferentialOperator& differentialOperator,
                   const BoundaryConditions& boundaryConditions,
                   const RightHandSide& rightHandSide,
                   DofVector& u,
                   DofVector& fu ) const;

      class TraverserBoundaryEntitiesProcessor
      {
         public:

            /****
             * TODO: This must be specialized for entities with different dimensions
             * otherwise 'coordinates' would not make sense without knowing the orientation.
             */
#ifdef HAVE_CUDA
            __host__ __device__
#endif
            static void processCell( const MeshType& mesh,
                                     TraverserUserData& userData,
                                     const IndexType index,
                                     const CoordinatesType& coordinates )
            {
               userData.boundaryConditions->setBoundaryConditions( *userData.time,
                                                                   mesh,
                                                                   index,
                                                                   coordinates,
                                                                   *userData.u,
                                                                   *userData.fu );
            }

#ifdef HAVE_CUDA
            __host__ __device__
#endif
            static void processFace( const MeshType& mesh,
                                     TraverserUserData& userData,
                                     const IndexType index,
                                     const CoordinatesType& coordinates )
            {
               userData.boundaryConditions->setBoundaryConditions( *userData.time,
                                                                   mesh,
                                                                   index,
                                                                   coordinates,
                                                                   *userData.u,
                                                                   *userData.fu );
            }


      };

      class TraverserInteriorEntitiesProcessor
      {
         public:

         typedef typename MeshType::VertexType VertexType;

#ifdef HAVE_CUDA
            __host__ __device__
#endif
            static void processCell( const MeshType& mesh,
                                     TraverserUserData& userData,
                                     const IndexType index,
                                     const CoordinatesType& coordinates )
            {
               ( *userData.fu)[ index ] = userData.differentialOperator->getValue( mesh,
                                                                                   index,
                                                                                   coordinates,
                                                                                   *userData.u,
                                                                                   *userData.time );

               typedef tnlFunctionAdapter< MeshType, RightHandSide > FunctionAdapter;
               ( * userData.fu )[ index ] += FunctionAdapter::getValue( mesh,
                                                                        *userData.rightHandSide,
                                                                        index,
                                                                        coordinates,
                                                                        *userData.time );
            }

#ifdef HAVE_CUDA
            __host__ __device__
#endif
            static void processFace( const MeshType& mesh,
                                     TraverserUserData& userData,
                                     const IndexType index,
                                     const CoordinatesType& coordinates )
            {
               ( *userData.fu)[ index ] = userData.differentialOperator->getValue( mesh,
                                                                                   index,
                                                                                   coordinates,
                                                                                   *userData.u,
                                                                                   *userData.time );

               typedef tnlFunctionAdapter< MeshType, RightHandSide > FunctionAdapter;
               ( * userData.fu )[ index ] += FunctionAdapter::getValue( mesh,
                                                                        *userData.rightHandSide,
                                                                        index,
                                                                        coordinates,
                                                                        *userData.time );
            }
      };

};

#include <solvers/pde/tnlExplicitUpdater_impl.h>
#endif /* TNLEXPLICITUPDATER_H_ */
