/***************************************************************************
                          tnlExplicitUpdater.h  -  description
                             -------------------
    begin                : Jul 29, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Functions/tnlFunctionAdapter.h>
#include <TNL/Timer.h>

namespace TNL {

template< typename Real,
          typename MeshFunction,
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

      MeshFunction *u, *fu;

      tnlExplicitUpdaterTraverserUserData( const Real& time,
                                           const DifferentialOperator& differentialOperator,
                                           const BoundaryConditions& boundaryConditions,
                                           const RightHandSide& rightHandSide,
                                           MeshFunction& u,
                                           MeshFunction& fu )
      : time( &time ),
        differentialOperator( &differentialOperator ),
        boundaryConditions( &boundaryConditions ),
        rightHandSide( &rightHandSide ),
        u( &u ),
        fu( &fu )
      {
      };
};


template< typename Mesh,
          typename MeshFunction,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide >
class tnlExplicitUpdater
{
   public:
      typedef Mesh MeshType;
      typedef typename MeshFunction::RealType RealType;
      typedef typename MeshFunction::DeviceType DeviceType;
      typedef typename MeshFunction::IndexType IndexType;
      typedef tnlExplicitUpdaterTraverserUserData< RealType,
                                                   MeshFunction,
                                                   DifferentialOperator,
                                                   BoundaryConditions,
                                                   RightHandSide > TraverserUserData;
 
      tnlExplicitUpdater()
      : gpuTransferTimer( 0 ){}
 
      void setGPUTransferTimer( tnlTimer& timer )
      {
         this->gpuTransferTimer = &timer;
      }

      template< typename EntityType >
      void update( const RealType& time,
                   const MeshType& mesh,
                   const DifferentialOperator& differentialOperator,
                   const BoundaryConditions& boundaryConditions,
                   const RightHandSide& rightHandSide,
                   MeshFunction& u,
                   MeshFunction& fu ) const;
 
      class TraverserBoundaryEntitiesProcessor
      {
         public:
 
            template< typename GridEntity >
            __cuda_callable__
            static inline void processEntity( const MeshType& mesh,
                                              TraverserUserData& userData,
                                              const GridEntity& entity )
            {
               ( *userData.u )( entity ) = userData.boundaryConditions->operator()
               ( *userData.u,
                 entity,
                 *userData.time );
            }

      };

      class TraverserInteriorEntitiesProcessor
      {
         public:

            typedef typename MeshType::VertexType VertexType;
 
            template< typename EntityType >
            __cuda_callable__
            static inline void processEntity( const MeshType& mesh,
                                              TraverserUserData& userData,
                                              const EntityType& entity )
            {
               ( *userData.fu)( entity ) =
                  userData.differentialOperator->operator()(
                     *userData.u,
                     entity,
                     *userData.time );

               typedef tnlFunctionAdapter< MeshType, RightHandSide > FunctionAdapter;
               ( * userData.fu )( entity ) +=
                  FunctionAdapter::getValue(
                     *userData.rightHandSide,
                     entity,
                     *userData.time );
            }
      };
 
   protected:
 
      tnlTimer* gpuTransferTimer;
};

} // namespace TNL

#include <TNL/solvers/pde/tnlExplicitUpdater_impl.h>

