/***************************************************************************
                          HeatEquationProblem.h  -  description
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

#include <TNL/Problems/PDEProblem.h>
#include <TNL/Operators/diffusion/LinearDiffusion.h>
#include <TNL/Functions/MeshFunctionView.h>
#include <TNL/Timer.h>
#include <TNL/Solvers/PDE/ExplicitUpdater.h>
#include <TNL/Solvers/PDE/LinearSystemAssembler.h>
#include <TNL/Solvers/PDE/BackwardTimeDiscretisation.h>

#include <TNL/Meshes/DistributedMeshes/DistributedMeshSynchronizer.h>

namespace TNL {
namespace Problems {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename DifferentialOperator = Operators::LinearDiffusion< Mesh,
                                                              typename BoundaryCondition::RealType > >
class HeatEquationProblem : public PDEProblem< Mesh,
                                               Communicator,
                                               typename Mesh::RealType,
                                               typename Mesh::DeviceType,
                                               typename Mesh::IndexType  >
{
   public:

      typedef typename Mesh::RealType RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef typename Mesh::IndexType IndexType;
      typedef Functions::MeshFunctionView< Mesh > MeshFunctionType;
      typedef Pointers::SharedPointer< MeshFunctionType, DeviceType > MeshFunctionPointer;
      typedef PDEProblem< Mesh, Communicator, RealType, DeviceType, IndexType > BaseType;
      typedef Pointers::SharedPointer<  DifferentialOperator > DifferentialOperatorPointer;
      typedef Pointers::SharedPointer<  BoundaryCondition > BoundaryConditionPointer;
      typedef Pointers::SharedPointer<  RightHandSide, DeviceType > RightHandSidePointer;

      using typename BaseType::MeshType;
      using typename BaseType::MeshPointer;
      using typename BaseType::DofVectorType;
      using typename BaseType::DofVectorPointer;
      using typename BaseType::MatrixType;

      typedef Communicator CommunicatorType;

      String getPrologHeader() const;

      void writeProlog( Logger& logger,
                        const Config::ParameterContainer& parameters ) const;

      bool writeEpilog( Logger& logger );


      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix );

      bool setInitialCondition( const Config::ParameterContainer& parameters,
                                DofVectorPointer& dofs );

      template< typename MatrixPointer >
      bool setupLinearSystem( MatrixPointer& matrixPointer );

      bool makeSnapshot( const RealType& time,
                         const IndexType& step,
                         DofVectorPointer& dofs );

      IndexType getDofs() const;

      void bindDofs( DofVectorPointer& dofs );

      void getExplicitUpdate( const RealType& time,
                              const RealType& tau,
                              DofVectorPointer& _u,
                              DofVectorPointer& _fu );

      void applyBoundaryConditions( const RealType& time,
                                    DofVectorPointer& dofs );

      template< typename MatrixPointer >
      void assemblyLinearSystem( const RealType& time,
                                 const RealType& tau,
                                 DofVectorPointer& dofsPointer,
                                 MatrixPointer& matrixPointer,
                                 DofVectorPointer& rightHandSidePointer );

   protected:

      using DistributedMeshSynchronizerType = Meshes::DistributedMeshes::DistributedMeshSynchronizer< Meshes::DistributedMeshes::DistributedMesh< typename MeshFunctionType::MeshType > >;
      DistributedMeshSynchronizerType synchronizer;

      MeshFunctionPointer uPointer;
      MeshFunctionPointer fuPointer;

      DifferentialOperatorPointer differentialOperatorPointer;

      BoundaryConditionPointer boundaryConditionPointer;

      RightHandSidePointer rightHandSidePointer;

      Timer gpuTransferTimer;

      Solvers::PDE::ExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;

      Solvers::PDE::LinearSystemAssembler< Mesh,
                                           MeshFunctionType,
                                           DifferentialOperator,
                                           BoundaryCondition,
                                           RightHandSide,
                                           Solvers::PDE::BackwardTimeDiscretisation,
                                           DofVectorType > systemAssembler;

     bool catchExceptions = true;
};

} // namespace Problems
} // namespace TNL

#include <TNL/Problems/HeatEquationProblem_impl.h>
