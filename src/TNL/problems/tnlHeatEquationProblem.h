/***************************************************************************
                          tnlHeatEquationProblem.h  -  description
                             -------------------
    begin                : Feb 23, 2013
    copyright            : (C) 2013 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Szekely Ondrej, ondra.szekely@gmail.com
 */

#pragma once

#include <TNL/problems/tnlPDEProblem.h>
#include <TNL/operators/diffusion/tnlLinearDiffusion.h>
#include <TNL/matrices/tnlEllpackMatrix.h>
#include <TNL/Functions/tnlMeshFunction.h>
#include <TNL/Timer.h>

namespace TNL {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator = tnlLinearDiffusion< Mesh,
                                                              typename BoundaryCondition::RealType > >
class tnlHeatEquationProblem : public tnlPDEProblem< Mesh,
                                                     typename DifferentialOperator::RealType,
                                                     typename Mesh::DeviceType,
                                                     typename DifferentialOperator::IndexType  >
{
   public:

      typedef typename DifferentialOperator::RealType RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef typename DifferentialOperator::IndexType IndexType;
      typedef Functions::tnlMeshFunction< Mesh > MeshFunctionType;
      typedef tnlSharedPointer< MeshFunctionType, DeviceType > MeshFunctionPointer;
      typedef tnlPDEProblem< Mesh, RealType, DeviceType, IndexType > BaseType;
      typedef tnlCSRMatrix< RealType, DeviceType, IndexType > MatrixType;
      typedef tnlSharedPointer< DifferentialOperator > DifferentialOperatorPointer;
      typedef tnlSharedPointer< BoundaryCondition > BoundaryConditionPointer;
      typedef tnlSharedPointer< RightHandSide, DeviceType > RightHandSidePointer;

      using typename BaseType::MeshType;
      using typename BaseType::MeshPointer;
      using typename BaseType::DofVectorType;
      using typename BaseType::DofVectorPointer;
      using typename BaseType::MeshDependentDataType;

      static String getTypeStatic();

      String getPrologHeader() const;

      void writeProlog( Logger& logger,
                        const Config::ParameterContainer& parameters ) const;
 
      bool writeEpilog( Logger& logger );


      bool setup( const Config::ParameterContainer& parameters );

      bool setInitialCondition( const Config::ParameterContainer& parameters,
                                const MeshPointer& mesh,
                                DofVectorPointer& dofs,
                                MeshDependentDataType& meshDependentData );

      template< typename MatrixPointer >
      bool setupLinearSystem( const MeshPointer& meshPointer,
                              MatrixPointer& matrixPointer );

      bool makeSnapshot( const RealType& time,
                         const IndexType& step,
                         const MeshPointer& meshPointer,
                         DofVectorPointer& dofs,
                         MeshDependentDataType& meshDependentData );

      IndexType getDofs( const MeshPointer& meshPointer ) const;

      void bindDofs( const MeshPointer& meshPointer,
                     const DofVectorPointer& dofs );

      void getExplicitRHS( const RealType& time,
                           const RealType& tau,
                           const MeshPointer& meshPointer,
                           DofVectorPointer& _u,
		           DofVectorPointer& _fu,
                           MeshDependentDataType& meshDependentData );

      template< typename MatrixPointer >
      void assemblyLinearSystem( const RealType& time,
                                 const RealType& tau,
                                 const MeshPointer& meshPointer,
                                 const DofVectorPointer& dofsPointer,
                                 MatrixPointer& matrixPointer,
                                 DofVectorPointer& rightHandSidePointer,
                                 MeshDependentDataType& meshDependentData );


      protected:
         
         MeshFunctionPointer uPointer;
      
         DifferentialOperatorPointer differentialOperatorPointer;

         BoundaryConditionPointer boundaryConditionPointer;

         RightHandSidePointer rightHandSidePointer;
         
         tnlTimer gpuTransferTimer;
};

} // namespace TNL

#include <TNL/problems/tnlHeatEquationProblem_impl.h>
