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
#include <TNL/Pointers/SharedPointer.h>
#include <type_traits>
#include "Traverser_Grid2D.h"

namespace TNL {

template< typename Real,
          typename MeshFunction,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide >
class ExplicitUpdaterTraverserUserData
{
   public:
      
      using RealType = Real;
      using MeshFunctionType = MeshFunction;
      using DifferentialOperatorType = DifferentialOperator;
      using BoundaryConditionsType = BoundaryConditions;
      using RightHandSideType = RightHandSide;
      
      Real time;

      const DifferentialOperator* differentialOperator;

      const BoundaryConditions* boundaryConditions;

      const RightHandSide* rightHandSide;

      //MeshFunction *uMf, *fuMf;
      
      Real *u, *fu;
      
      ExplicitUpdaterTraverserUserData()
      : time( 0.0 ),
        differentialOperator( NULL ),
        boundaryConditions( NULL ),
        rightHandSide( NULL ),
        u( NULL ),
        fu( NULL )
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
      typedef Pointers::SharedPointer<  MeshType > MeshPointer;
      typedef typename MeshFunction::RealType RealType;
      typedef typename MeshFunction::DeviceType DeviceType;
      typedef typename MeshFunction::IndexType IndexType;
      typedef ExplicitUpdaterTraverserUserData< RealType,
                                                MeshFunction,
                                                DifferentialOperator,
                                                BoundaryConditions,
                                                RightHandSide > TraverserUserData;
      typedef Pointers::SharedPointer<  DifferentialOperator, DeviceType > DifferentialOperatorPointer;
      typedef Pointers::SharedPointer<  BoundaryConditions, DeviceType > BoundaryConditionsPointer;
      typedef Pointers::SharedPointer<  RightHandSide, DeviceType > RightHandSidePointer;
      typedef Pointers::SharedPointer<  MeshFunction, DeviceType > MeshFunctionPointer;
      typedef Pointers::SharedPointer<  TraverserUserData, DeviceType > TraverserUserDataPointer;
      
      void setDifferentialOperator( const DifferentialOperatorPointer& differentialOperatorPointer )
      {
         this->userDataPointer->differentialOperator = &differentialOperatorPointer.template getData< DeviceType >();
      }
      
      void setBoundaryConditions( const BoundaryConditionsPointer& boundaryConditionsPointer )
      {
         this->userDataPointer->boundaryConditions = &boundaryConditionsPointer.template getData< DeviceType >();
      }
      
      void setRightHandSide( const RightHandSidePointer& rightHandSidePointer )
      {
         this->userDataPointer->rightHandSide = &rightHandSidePointer.template getData< DeviceType >();
      }
            
      template< typename EntityType >
      void update( const RealType& time,
                   const RealType& tau,
                   const MeshPointer& meshPointer,
                   MeshFunctionPointer& uPointer,
                   MeshFunctionPointer& fuPointer )
      {
         static_assert( std::is_same< MeshFunction,
                                      Containers::Vector< typename MeshFunction::RealType,
                                                 typename MeshFunction::DeviceType,
                                                 typename MeshFunction::IndexType > >::value != true,
            "Error: I am getting Vector instead of MeshFunction or similar object. You might forget to bind DofVector into MeshFunction in you method getExplicitUpdate."  );
            
         TNL_ASSERT_TRUE( this->userDataPointer->differentialOperator,
                          "The differential operator is not correctly set-up. Use method setDifferentialOperator() to do it." );
         TNL_ASSERT_TRUE( this->userDataPointer->boundaryConditions, 
                          "The boundary conditions are not correctly set-up. Use method setBoundaryCondtions() to do it." );
         TNL_ASSERT_TRUE( this->userDataPointer->rightHandSide, 
                          "The right-hand side is not correctly set-up. Use method setRightHandSide() to do it." );
         
         
         this->userDataPointer->time = time;
         this->userDataPointer->u = uPointer->getData().getData();
         this->userDataPointer->fu = fuPointer->getData().getData();
         TNL::Traverser< MeshType, EntityType > meshTraverser;
         meshTraverser.template processInteriorEntities< TraverserUserData,
                                                         TraverserInteriorEntitiesProcessor >
                                                       ( meshPointer,
                                                         userDataPointer );
         this->userDataPointer->time = time + tau;
         /*meshTraverser.template processBoundaryEntities< TraverserUserData,
                                             TraverserBoundaryEntitiesProcessor >
                                           ( meshPointer,
                                             userDataPointer );*/  
      }
      
      template< typename EntityType >
      void applyBoundaryConditions( const MeshPointer& meshPointer,
                                    const RealType& time,
                                    MeshFunctionPointer& uPointer )
      {
         this->userDataPointer->time = time;
         this->userDataPointer->u = uPointer->getData().getData();
         Meshes::Traverser< MeshType, EntityType > meshTraverser;
         meshTraverser.template processBoundaryEntities< TraverserUserData,
                                             TraverserBoundaryEntitiesProcessor >
                                           ( meshPointer,
                                             *userDataPointer );         
      }      
      
         
      class TraverserBoundaryEntitiesProcessor
      {
         public:

            template< typename GridEntity >
            __cuda_callable__
            static inline void processEntity( const MeshType& mesh,
                                              TraverserUserData& userData,
                                              const GridEntity& entity )
            {
               /*( *userData.u )( entity ) = ( *userData.boundaryConditions )
                  ( *userData.u, entity, userData.time );*/
            }
            
            //template< typename EntityType >            
            __cuda_callable__
            static inline void processEntity( const MeshType& mesh,
                                              TraverserUserData& userData,
                                              //const EntityType& entity,
                                              const IndexType& entityIndex,
                                              const typename MeshType::CoordinatesType& coordinates )
            {
               userData.u[ entityIndex ] = 0.0; /*( *userData.boundaryConditions )
                  ( *userData.u, entity, userData.time );*/
            }
            
      };
      

      class TraverserInteriorEntitiesProcessor
      {
         public:

            typedef typename MeshType::PointType PointType;

            template< typename EntityType >
            __cuda_callable__
            static inline void processEntity( const MeshType& mesh,
                                              TraverserUserData& userData,
                                              const EntityType& entity )
            {
               typedef Functions::FunctionAdapter< MeshType, RightHandSide > FunctionAdapter;
               ( userData.fu )[ entity.getIndex() ]  = 
                  ( *userData.differentialOperator )( userData.u, entity, userData.time );
                   + FunctionAdapter::getValue( *userData.rightHandSide, entity, userData.time );
               
            }

            //template< typename EntityType >
            __cuda_callable__
            static inline void processEntity( const MeshType& mesh,
                                              TraverserUserData& userData,
                                              //const EntityType& entity,
                                              const IndexType& entityIndex,
                                              const typename MeshType::CoordinatesType& coordinates )
            {
               typedef Functions::FunctionAdapter< MeshType, RightHandSide > FunctionAdapter;
               userData.fu[ entityIndex ] = 
                       ( *userData.differentialOperator )( mesh, userData.u, entityIndex, coordinates, userData.time );
                    //   + 0.0;
            }
            
      }; 

   protected:

      TraverserUserDataPointer userDataPointer;

};

} // namepsace TNL

