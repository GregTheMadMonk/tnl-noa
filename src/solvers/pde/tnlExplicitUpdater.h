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

#include <functions/tnlFunctionAdapter.h>
#include <core/tnlTimer.h>
#include <core/tnlDevice_Callable.h>

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
      {};
           
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

#include <solvers/pde/tnlExplicitUpdater_impl.h>
#endif /* TNLEXPLICITUPDATER_H_ */
