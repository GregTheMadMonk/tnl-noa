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
      
      typedef typename DifferentialOperator::DeviceType DeviceType;
      typedef tnlSharedPointer< DifferentialOperator, DeviceType > DifferentialOperatorPointer;
      typedef tnlSharedPointer< BoundaryConditions, DeviceType > BoundaryConditionsPointer;
      typedef tnlSharedPointer< RightHandSide, DeviceType > RightHandSidePointer;
      typedef tnlSharedPointer< MeshFunction, DeviceType > MeshFunctionPointer;

      const DifferentialOperatorPointer differentialOperator;

      const BoundaryConditionsPointer boundaryConditions;

      const RightHandSidePointer rightHandSide;

      MeshFunctionPointer u, fu;
      
      public:

         const Real time;         


      tnlExplicitUpdaterTraverserUserData( const Real& time,
                                           const DifferentialOperatorPointer& differentialOperator,
                                           const BoundaryConditionsPointer& boundaryConditions,
                                           const RightHandSidePointer& rightHandSide,
                                           MeshFunctionPointer& u,
                                           MeshFunctionPointer& fu )
      : time( time ),
        differentialOperator( differentialOperator ),
        boundaryConditions( boundaryConditions ),
        rightHandSide( rightHandSide ),
        u( u ),
        fu( fu )
      {
      };
      
      /*DifferentialOperator& differentialOperator()
      {
         return this->differentialOperator; //* ( DifferentialOperator* ) data;
      }
      
      BoundaryConditions& boundaryConditions()
      {
         return this->boundaryConditions; //* ( BoundaryConditions* ) & data[ sizeof( DifferentialOperator ) ];
      }
      
      RightHandSide& rightHandSide()
      {
         return this->rightHandSide; //* ( RightHandSide* ) & data[ sizeof( DifferentialOperator ) +
                                     //        sizeof( BoundaryConditions ) ];
      }
      
      MeshFunction& u()
      {
         return this->u; //* ( MeshFunction* ) & data[ sizeof( DifferentialOperator ) +
                         //                   sizeof( BoundaryConditions ) + 
                         //                   sizeof( RightHandSide )];
      }
      
      MeshFunction& fu()
      {
         return this->fu; //* ( MeshFunction* ) & data[ sizeof( DifferentialOperator ) +
                          //                  sizeof( BoundaryConditions ) + 
                          //                  sizeof( RightHandSide ) + 
                          //                  sizeof( MeshFunction ) ];
      }*/
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
               printf( "BC size %d ptr %p \n", sizeof( BoundaryConditions ), &userData.boundaryConditions.template getData< DeviceType >() );
               ( userData.u.template modifyData< DeviceType >() )( entity ) = ( userData.boundaryConditions.template getData< DeviceType >() )
               ( userData.u.template getData< DeviceType >(),
                 entity,
                 userData.time );
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
               printf( "DIF.OP. size %d ptr %p \n", sizeof( DifferentialOperator ), &userData.differentialOperator.template getData< DeviceType >() );
               printf( "RHS. size %d ptr %p \n", sizeof( RightHandSide ), &userData.rightHandSide.template getData< DeviceType >() );
               /*( *userData.fu )( entity ) =
                  ( userData.differentialOperatorPointer.template getData< DeviceType >() )(
                     *userData.u, 
                     entity,
                     *userData.time );

               typedef tnlFunctionAdapter< MeshType, RightHandSide > FunctionAdapter;
               (  *userData.fu )( entity ) += 
                  FunctionAdapter::getValue(
                     userData.rightHandSidePointer.template getData< DeviceType >(),
                     entity,
                     *userData.time );
                */
            }
      };
      
   protected:
      
      tnlTimer* gpuTransferTimer;
};

#include <solvers/pde/tnlExplicitUpdater_impl.h>
#endif /* TNLEXPLICITUPDATER_H_ */
