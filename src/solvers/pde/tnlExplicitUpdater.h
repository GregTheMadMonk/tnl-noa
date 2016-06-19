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
#include <core/tnlSharedPointer.h>

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

               typedef tnlFunctionAdapter< MeshType, RightHandSide > FunctionAdapter;
               (  *userData.fu )( entity ) += 
                  FunctionAdapter::getValue( *userData.rightHandSide, entity, userData.time );
            }
      };
      
   protected:
      
      tnlTimer* gpuTransferTimer;
};

#include <solvers/pde/tnlExplicitUpdater_impl.h>
#endif /* TNLEXPLICITUPDATER_H_ */
