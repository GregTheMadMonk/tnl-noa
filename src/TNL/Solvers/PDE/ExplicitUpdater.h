/***************************************************************************
                          ExplicitUpdater.h  -  description
                             -------------------
    begin                : Jul 29, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Functions/FunctionAdapter.h>
#include <TNL/Timer.h>
#include <TNL/SharedPointer.h>

namespace TNL {
namespace Solvers {
namespace PDE {   

template< typename Real,
          typename MeshFunction,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide >
class ExplicitUpdaterTraverserUserData
{
   public:
      
      const Real time;

      const DifferentialOperator* differentialOperator;

      const BoundaryConditions* boundaryConditions;

      const RightHandSide* rightHandSide;

      MeshFunction *u, *fu;
      
      ExplicitUpdaterTraverserUserData( const Real& time,
                                        const DifferentialOperator* differentialOperator,
                                        const BoundaryConditions* boundaryConditions,
                                        const RightHandSide* rightHandSide,
                                        MeshFunction* u,
                                        MeshFunction* fu )
      : time( time ),
        differentialOperator( differentialOperator ),
        boundaryConditions( boundaryConditions ),
        rightHandSide( rightHandSide ),
        u( u ),
        fu( fu )
      {}
};


template< typename Mesh,
          typename MeshFunction,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide >
class ExplicitUpdater
{
   public:
      typedef Mesh MeshType;
      typedef SharedPointer< MeshType > MeshPointer;
      typedef typename MeshFunction::RealType RealType;
      typedef typename MeshFunction::DeviceType DeviceType;
      typedef typename MeshFunction::IndexType IndexType;
      typedef ExplicitUpdaterTraverserUserData< RealType,
                                                   MeshFunction,
                                                   DifferentialOperator,
                                                   BoundaryConditions,
                                                   RightHandSide > TraverserUserData;
      typedef SharedPointer< DifferentialOperator, DeviceType > DifferentialOperatorPointer;
      typedef SharedPointer< BoundaryConditions, DeviceType > BoundaryConditionsPointer;
      typedef SharedPointer< RightHandSide, DeviceType > RightHandSidePointer;
      typedef SharedPointer< MeshFunction, DeviceType > MeshFunctionPointer;
      
      ExplicitUpdater()
      : gpuTransferTimer( 0 ){}
 
      void setGPUTransferTimer( Timer& timer )
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

               typedef Functions::FunctionAdapter< MeshType, RightHandSide > FunctionAdapter;
               (  *userData.fu )( entity ) += 
                  FunctionAdapter::getValue( *userData.rightHandSide, entity, userData.time );
            }
      };
 
   protected:
 
      Timer* gpuTransferTimer;
};

} // namespace PDE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/PDE/ExplicitUpdater_impl.h>

