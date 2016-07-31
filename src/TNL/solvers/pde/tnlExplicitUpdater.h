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
#include <TNL/tnlSharedPointer.h>

namespace TNL {

template< typename Real,
          typename MeshFunction,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide >
class tnlExplicitUpdaterTraverserUserData
{
   public:
      
      /*typedef typename DifferentialOperator::DeviceType DeviceType;
      /*typedef DifferentialOperator DifferentialOperator;
      typedef BoundaryConditions BoundaryConditions;
      typedef RightHandSide RightHandSide;
      typedef MeshFunction MeshFunction;*/

      const DifferentialOperator* differentialOperator;

      const BoundaryConditions* boundaryConditions;

      const RightHandSide* rightHandSide;

      MeshFunction *u, *fu;
      
      Real* _fu;
      
      public:

         const Real time;         


      tnlExplicitUpdaterTraverserUserData( const Real& time,
                                           const DifferentialOperator& differentialOperator,
                                           const BoundaryConditions& boundaryConditions,
                                           const RightHandSide& rightHandSide,
                                           MeshFunction& u,
                                           MeshFunction& fu,
                                           MeshFunction& __fu )
      : time( time ),
        differentialOperator( &differentialOperator ),
        boundaryConditions( &boundaryConditions ),
        rightHandSide( &rightHandSide ),
        u( &u ),
        fu( &fu ),
        _fu( &__fu[ 0 ] )
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
      typedef tnlSharedPointer< MeshType > MeshPointer;
      typedef typename MeshFunction::RealType RealType;
      typedef typename MeshFunction::DeviceType DeviceType;
      typedef typename MeshFunction::IndexType IndexType;
      typedef tnlExplicitUpdaterTraverserUserData< RealType,
                                                   MeshFunction,
                                                   DifferentialOperator,
                                                   BoundaryConditions,
                                                   RightHandSide > TraverserUserData;
      typedef tnlSharedPointer< DifferentialOperator, DeviceType > DifferentialOperatorPointer;
      typedef tnlSharedPointer< BoundaryConditions, DeviceType > BoundaryConditionsPointer;
      typedef tnlSharedPointer< RightHandSide, DeviceType > RightHandSidePointer;
      typedef tnlSharedPointer< MeshFunction, DeviceType > MeshFunctionPointer;
      
      tnlExplicitUpdater()
      : gpuTransferTimer( 0 ){}
 
      void setGPUTransferTimer( tnlTimer& timer )
      {
         this->gpuTransferTimer = &timer;
      }

      template< typename EntityType >
      void update( const RealType& time,
                   const MeshPointer& meshPointer,
                   const DifferentialOperatorPointer& differentialOperatorPointer,
                   const BoundaryConditionsPointer& boundaryConditionsPointer,
                   const RightHandSidePointer& rightHandSidePointer,
                   MeshFunctionPointer& uPointer,
                   MeshFunctionPointer& fuPointer ) const;      
      
      class TraverserBoundaryEntitiesProcessor
      {
         public:
 
            template< typename GridEntity >
            __cuda_callable__
            static inline void processEntity( const MeshType& mesh,
                                              TraverserUserData& userData,
                                              const GridEntity& entity )
            {
               ( *userData.u )( entity ) = ( *userData.boundaryConditions )
                  ( *userData.u, entity, userData.time );
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
               ( *userData.fu )( entity ) =                
                  ( *userData.differentialOperator )( *userData.u, entity, userData.time );

               typedef Functions::tnlFunctionAdapter< MeshType, RightHandSide > FunctionAdapter;
               (  *userData.fu )( entity ) += 
                  FunctionAdapter::getValue( *userData.rightHandSide, entity, userData.time );
            }
      };
 
   protected:
 
      tnlTimer* gpuTransferTimer;
};

} // namespace TNL

#include <TNL/solvers/pde/tnlExplicitUpdater_impl.h>

