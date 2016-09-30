/***************************************************************************
                          LinearSystemAssembler.h  -  description
                             -------------------
    begin                : Oct 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/SharedPointer.h>
#include <TNL/Functions/FunctionAdapter.h>

namespace TNL {
namespace Solvers {
namespace PDE {   

template< typename Real,
          typename MeshFunction,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix,
          typename DofVector >
class LinearSystemAssemblerTraverserUserData
{
   public:
      typedef Matrix MatrixType;
      typedef typename Matrix::DeviceType DeviceType;

      const Real time;

      const Real tau;

      const DifferentialOperator* differentialOperator;

      const BoundaryConditions* boundaryConditions;

      const RightHandSide* rightHandSide;
      
      const MeshFunction* u;
      
      DofVector* b;

      Matrix* matrix;

      LinearSystemAssemblerTraverserUserData( const Real& time,
                                              const Real& tau,
                                              const DifferentialOperator* differentialOperator,
                                              const BoundaryConditions* boundaryConditions,
                                              const RightHandSide* rightHandSide,
                                              const MeshFunction* u,
                                              Matrix* matrix,
                                              DofVector* b )
      : time( time ),
        tau( tau ),
        differentialOperator( differentialOperator ),
        boundaryConditions( boundaryConditions ),
        rightHandSide( rightHandSide ),
        u( u ),
        b( b ),
        matrix( matrix )
      {}

   protected:

};


template< typename Mesh,
          typename MeshFunction,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename TimeDiscretisation,
          typename Matrix,
          typename DofVector >
class LinearSystemAssembler
{
   public:
   typedef typename MeshFunction::MeshType MeshType;
   typedef typename MeshFunction::MeshPointer MeshPointer;
   typedef typename MeshFunction::RealType RealType;
   typedef typename MeshFunction::DeviceType DeviceType;
   typedef typename MeshFunction::IndexType IndexType;
   typedef Matrix MatrixType;
   typedef LinearSystemAssemblerTraverserUserData< RealType,
                                                      MeshFunction,
                                                      DifferentialOperator,
                                                      BoundaryConditions,
                                                      RightHandSide,
                                                      MatrixType,
                                                      DofVector > TraverserUserData;

   typedef SharedPointer< Matrix, DeviceType > MatrixPointer;
   typedef SharedPointer< DifferentialOperator, DeviceType > DifferentialOperatorPointer;
   typedef SharedPointer< BoundaryConditions, DeviceType > BoundaryConditionsPointer;
   typedef SharedPointer< RightHandSide, DeviceType > RightHandSidePointer;
   typedef SharedPointer< MeshFunction, DeviceType > MeshFunctionPointer;
   typedef SharedPointer< DofVector, DeviceType > DofVectorPointer;
   
      
   template< typename EntityType >
   void assembly( const RealType& time,
                  const RealType& tau,
                  const MeshPointer& meshPointer,
                  const DifferentialOperatorPointer& differentialOperatorPointer,
                  const BoundaryConditionsPointer& boundaryConditionsPointer,
                  const RightHandSidePointer& rightHandSidePointer,
                  const MeshFunctionPointer& uPointer,
                  MatrixPointer& matrixPointer,
                  DofVectorPointer& bPointer ) const;

 
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
                 ( *userData.matrix ),
                 ( *userData.b ) );
         }
   };

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
                 ( *userData.matrix ),
                 ( *userData.b ) );
 
            typedef Functions::FunctionAdapter< MeshType, RightHandSide > RhsFunctionAdapter;
            typedef Functions::FunctionAdapter< MeshType, MeshFunction > MeshFunctionAdapter;
            const RealType& rhs = RhsFunctionAdapter::getValue
               ( ( *userData.rightHandSide ),
                 entity,
                 userData.time );
            TimeDiscretisation::applyTimeDiscretisation( ( *userData.matrix ),
                                                         ( *userData.b )[ entity.getIndex() ],
                                                         entity.getIndex(),
                                                         MeshFunctionAdapter::getValue( ( *userData.u ), entity, userData.time ),
                                                         userData.tau,
                                                         rhs );
         }
   };
};

} // namespace PDE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/PDE/LinearSystemAssembler_impl.h>
