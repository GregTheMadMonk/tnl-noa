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
#include <type_traits>
#include <TNL/Meshes/GridDetails/Traverser_Grid1D.h>
#include <TNL/Meshes/GridDetails/Traverser_Grid2D.h>
#include <TNL/Meshes/GridDetails/Traverser_Grid3D.h>
#include <TNL/Solvers/PDE/ExplicitUpdater.h>


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
      
      Real time;

      const DifferentialOperator* differentialOperator;

      const BoundaryConditions* boundaryConditions;

      const RightHandSide* rightHandSide;

      MeshFunction *u, *fu;
      
      ExplicitUpdaterTraverserUserData()
      : time( 0.0 ),
        differentialOperator( NULL ),
        boundaryConditions( NULL ),
        rightHandSide( NULL ),
        u( NULL ),
        fu( NULL )
      {}
      
      
      /*void setUserData( const Real& time,
                        const DifferentialOperator* differentialOperator,
                        const BoundaryConditions* boundaryConditions,
                        const RightHandSide* rightHandSide,
                        MeshFunction* u,
                        MeshFunction* fu )
      {
         this->time = time;
         this->differentialOperator = differentialOperator;
         this->boundaryConditions = boundaryConditions;
         this->rightHandSide = rightHandSide;
         this->u = u;
         this->fu = fu;
      }*/
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
      typedef SharedPointer< TraverserUserData, DeviceType > TraverserUserDataPointer;
      
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
         this->userDataPointer->u = &uPointer.template modifyData< DeviceType >();
         this->userDataPointer->fu = &fuPointer.template modifyData< DeviceType >();
         Meshes::Traverser< MeshType, EntityType > meshTraverser;
         meshTraverser.template processInteriorEntities< TraverserUserData,
                                                         TraverserInteriorEntitiesProcessor >
                                                       ( meshPointer,
                                                         userDataPointer );
         this->userDataPointer->time = time + tau;
         meshTraverser.template processBoundaryEntities< TraverserUserData,
                                             TraverserBoundaryEntitiesProcessor >
                                           ( meshPointer,
                                             userDataPointer );
         
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
               ( *userData.u )( entity ) = ( *userData.boundaryConditions )
                  ( *userData.u, entity, userData.time );
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
           /*    std::cerr<<"===========================================================" << std::endl; 
               std::cerr<<"fu:" << userData.fu << std::endl; 
               std::cerr<< "diffOp:" << userData.differentialOperator << std::endl; 
               std::cerr<<"===========================================================" << std::endl; 
               
               std::cerr<<std::flush;*/
               
            //   int blabla;
             //  std::cin >> blabla; 
               
               ( *userData.fu )( entity ) = 
                       ( *userData.differentialOperator )( *userData.u, entity, userData.time );
            
               typedef Functions::FunctionAdapter< MeshType, RightHandSide > FunctionAdapter;
               (  *userData.fu )( entity ) += 
                  FunctionAdapter::getValue( *userData.rightHandSide, entity, userData.time );
               
            }
      }; 

   protected:

      TraverserUserDataPointer userDataPointer;

};

} // namespace PDE
} // namespace Solvers
} // namespace TNL

