/***************************************************************************
                          LinearSystemAssembler.h  -  description
                             -------------------
    begin                : Oct 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Functions/FunctionAdapter.h>
#include <TNL/Meshes/Traverser.h>

namespace TNL {
namespace Solvers {
namespace PDE {

template< typename Real,
          typename MeshFunction,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename DofVector >
class LinearSystemAssemblerTraverserUserData
{
   public:
      Real time = 0.0;

      Real tau = 0.0;

      const DifferentialOperator* differentialOperator = NULL;

      const BoundaryConditions* boundaryConditions = NULL;

      const RightHandSide* rightHandSide = NULL;

      const MeshFunction* u = NULL;

      DofVector* b = NULL;

      void* matrix = NULL;

      LinearSystemAssemblerTraverserUserData()
      : time( 0.0 ),
        tau( 0.0 ),
        differentialOperator( NULL ),
        boundaryConditions( NULL ),
        rightHandSide( NULL ),
        u( NULL ),
        b( NULL ),
        matrix( NULL )
      {}
};


template< typename Mesh,
          typename MeshFunction,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename TimeDiscretisation,
          typename DofVector >
class LinearSystemAssembler
{
   public:
   typedef typename MeshFunction::MeshType MeshType;
   typedef typename MeshFunction::MeshPointer MeshPointer;
   typedef typename MeshFunction::RealType RealType;
   typedef typename MeshFunction::DeviceType DeviceType;
   typedef typename MeshFunction::IndexType IndexType;
   typedef LinearSystemAssemblerTraverserUserData< RealType,
                                                   MeshFunction,
                                                   DifferentialOperator,
                                                   BoundaryConditions,
                                                   RightHandSide,
                                                   DofVector > TraverserUserData;

   //typedef Pointers::SharedPointer<  Matrix, DeviceType > MatrixPointer;
   typedef Pointers::SharedPointer<  DifferentialOperator, DeviceType > DifferentialOperatorPointer;
   typedef Pointers::SharedPointer<  BoundaryConditions, DeviceType > BoundaryConditionsPointer;
   typedef Pointers::SharedPointer<  RightHandSide, DeviceType > RightHandSidePointer;
   typedef Pointers::SharedPointer<  MeshFunction, DeviceType > MeshFunctionPointer;
   typedef Pointers::SharedPointer<  DofVector, DeviceType > DofVectorPointer;

   void setDifferentialOperator( const DifferentialOperatorPointer& differentialOperatorPointer )
   {
      this->userData.differentialOperator = &differentialOperatorPointer.template getData< DeviceType >();
   }

   void setBoundaryConditions( const BoundaryConditionsPointer& boundaryConditionsPointer )
   {
      this->userData.boundaryConditions = &boundaryConditionsPointer.template getData< DeviceType >();
   }

   void setRightHandSide( const RightHandSidePointer& rightHandSidePointer )
   {
      this->userData.rightHandSide = &rightHandSidePointer.template getData< DeviceType >();
   }

   template< typename EntityType, typename Matrix >
   void assembly( const RealType& time,
                  const RealType& tau,
                  const MeshPointer& meshPointer,
                  const MeshFunctionPointer& uPointer,
                  std::shared_ptr< Matrix >& matrixPointer,
                  DofVectorPointer& bPointer )
   {
      static_assert( std::is_same< MeshFunction,
                                Containers::Vector< typename MeshFunction::RealType,
                                           typename MeshFunction::DeviceType,
                                           typename MeshFunction::IndexType > >::value != true,
      "Error: I am getting Vector instead of MeshFunction or similar object. You might forget to bind DofVector into MeshFunction in you method getExplicitUpdate."  );

      //const IndexType maxRowLength = matrixPointer.template getData< Devices::Host >().getMaxRowLength();
      //TNL_ASSERT_GT( maxRowLength, 0, "maximum row length must be positive" );
      this->userData.time = time;
      this->userData.tau = tau;
      this->userData.u = &uPointer.template getData< DeviceType >();
      this->userData.matrix = ( void* ) &matrixPointer->getView();
      this->userData.b = &bPointer.template modifyData< DeviceType >();
      Meshes::Traverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserBoundaryEntitiesProcessor< Matrix> >
                                                    ( meshPointer,
                                                      userData );
      meshTraverser.template processInteriorEntities< TraverserInteriorEntitiesProcessor< Matrix > >
                                                    ( meshPointer,
                                                      userData );

   }

   template< typename Matrix >
   class TraverserBoundaryEntitiesProcessor
   {
      public:

         template< typename EntityType >
         __cuda_callable__
         static void processEntity( const MeshType& mesh,
                                    TraverserUserData& userData,
                                    const EntityType& entity )
         {
            ( *userData.b )[ entity.getIndex() ] = 0.0;
            userData.boundaryConditions->setMatrixElements(
                 ( *userData.u ),
                 entity,
                 userData.time + userData.tau,
                 userData.tau,
                 ( * ( Matrix* ) ( userData.matrix ) ),
                 ( *userData.b ) );
         }
   };

   template< typename Matrix >
   class TraverserInteriorEntitiesProcessor
   {
      public:

         template< typename EntityType >
         __cuda_callable__
         static void processEntity( const MeshType& mesh,
                                    TraverserUserData& userData,
                                    const EntityType& entity )
         {
            ( *userData.b )[ entity.getIndex() ] = 0.0;
            userData.differentialOperator->setMatrixElements(
                 ( *userData.u ),
                 entity,
                 userData.time + userData.tau,
                 userData.tau,
                 ( *( Matrix* )( userData.matrix ) ),
                 ( *userData.b ) );

            typedef Functions::FunctionAdapter< MeshType, RightHandSide > RhsFunctionAdapter;
            typedef Functions::FunctionAdapter< MeshType, MeshFunction > MeshFunctionAdapter;
            const RealType& rhs = RhsFunctionAdapter::getValue
               ( ( *userData.rightHandSide ),
                 entity,
                 userData.time );
            TimeDiscretisation::applyTimeDiscretisation( ( *( Matrix* )( userData.matrix ) ),
                                                         ( *userData.b )[ entity.getIndex() ],
                                                         entity.getIndex(),
                                                         MeshFunctionAdapter::getValue( ( *userData.u ), entity, userData.time ),
                                                         userData.tau,
                                                         rhs );
         }
   };

protected:
   TraverserUserData userData;
};

} // namespace PDE
} // namespace Solvers
} // namespace TNL
