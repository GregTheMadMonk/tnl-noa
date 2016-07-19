/***************************************************************************
                          tnlHeatEquationProblem_impl.h  -  description
                             -------------------
    begin                : Mar 10, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <core/mfilename.h>
#include <matrices/tnlMatrixSetter.h>
#include <matrices/tnlMultidiagonalMatrixSetter.h>
#include <core/tnlLogger.h>
#include <solvers/pde/tnlExplicitUpdater.h>
#include <solvers/pde/tnlBoundaryConditionsSetter.h>
#include <solvers/pde/tnlLinearSystemAssembler.h>
#include <solvers/pde/tnlBackwardTimeDiscretisation.h>

#include "tnlMeanCurvatureFlowProblem.h"

namespace TNL {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
tnlString
tnlMeanCurvatureFlowProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getTypeStatic()
{
   return tnlString( "tnlMeanCurvativeFlowProblem< " ) + Mesh :: getTypeStatic() + " >";
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
tnlString
tnlMeanCurvatureFlowProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getPrologHeader() const
{
   return tnlString( "Mean Curvative Flow" );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
tnlMeanCurvatureFlowProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
writeProlog( tnlLogger& logger, const tnlParameterContainer& parameters ) const
{
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
tnlMeanCurvatureFlowProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setup( const tnlParameterContainer& parameters )
{
   if( ! this->boundaryCondition.setup( parameters, "boundary-conditions-" ) ||
       ! this->rightHandSide.setup( parameters, "right-hand-side-" ) )
      return false;
   this->differentialOperator.nonlinearDiffusionOperator.operatorQ.setEps( parameters.getParameter< double >( "eps" ) );
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
typename tnlMeanCurvatureFlowProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::IndexType
tnlMeanCurvatureFlowProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getDofs( const MeshType& mesh ) const
{
   /****
    * Set-up DOFs and supporting grid functions
    */
   return mesh.template getEntitiesCount< typename Mesh::Cell >();
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
tnlMeanCurvatureFlowProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
bindDofs( const MeshType& mesh,
          DofVectorType& dofVector )
{
   const IndexType dofs = mesh.template getEntitiesCount< typename Mesh::Cell >();
   this->solution.bind( dofVector.getData(), dofs );
   //differentialOperator.nonlinearDiffusionOperator.operatorQ.bind(solution);
//   this->differentialOperator.setupDofs(mesh);
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
tnlMeanCurvatureFlowProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setInitialCondition( const tnlParameterContainer& parameters,
                     const MeshType& mesh,
                     DofVectorType& dofs,
                     MeshDependentDataType& meshDependentData )
{
   this->bindDofs( mesh, dofs );
   const tnlString& initialConditionFile = parameters.getParameter< tnlString >( "initial-condition" );
   if( ! this->solution.load( initialConditionFile ) )
   {
      std::cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." << std::endl;
      return false;
   }
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
template< typename Matrix >
bool
tnlMeanCurvatureFlowProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setupLinearSystem( const MeshType& mesh,
                   Matrix& matrix )
{
   const IndexType dofs = this->getDofs( mesh );
   typedef typename MatrixType::CompressedRowsLengthsVector CompressedRowsLengthsVectorType;
   CompressedRowsLengthsVectorType rowLengths;
   if( ! rowLengths.setSize( dofs ) )
      return false;
   tnlMatrixSetter< MeshType, DifferentialOperator, BoundaryCondition, CompressedRowsLengthsVectorType > matrixSetter;
   matrixSetter.template getCompressedRowsLengths< typename Mesh::Cell >(
      mesh,
      differentialOperator,
      boundaryCondition,
      rowLengths
   );
   matrix.setDimensions( dofs, dofs );
   if( ! matrix.setCompressedRowsLengths( rowLengths ) )
      return false;
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
tnlMeanCurvatureFlowProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
makeSnapshot( const RealType& time,
              const IndexType& step,
              const MeshType& mesh,
              DofVectorType& dofs,
              MeshDependentDataType& meshDependentData )
{
  std::cout << std::endl << "Writing output at time " << time << " step " << step << "." << std::endl;

   this->bindDofs( mesh, dofs );
   //cout << "dofs = " << dofs << std::endl;
   tnlString fileName;
   FileNameBaseNumberEnding( "u-", step, 5, ".tnl", fileName );
   if( ! this->solution.save( fileName ) )
      return false;
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
tnlMeanCurvatureFlowProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getExplicitRHS( const RealType& time,
                const RealType& tau,
                const MeshType& mesh,
                DofVectorType& inDofs,
                DofVectorType& outDofs,
		MeshDependentDataType& meshDependentData )
{
   /****
    * If you use an explicit solver like tnlEulerSolver or tnlMersonSolver, you
    * need to implement this method. Compute the right-hand side of
    *
    *   d/dt u(x) = fu( x, u )
    *
    * You may use supporting vectors again if you need.
    */

//   this->differentialOperator.computeFirstGradient(mesh,time,u);
 
   //cout << "u = " << u << std::endl;
   //this->bindDofs( mesh, u );
   MeshFunctionType u( mesh, inDofs );
   MeshFunctionType fu( mesh, outDofs );
   //differentialOperator.nonlinearDiffusionOperator.operatorQ.update( mesh, time );
   tnlExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;
   explicitUpdater.template update< typename Mesh::Cell >(
      time,
      mesh,
      this->differentialOperator,
      this->boundaryCondition,
      this->rightHandSide,
      u,
      fu );
 
   tnlBoundaryConditionsSetter< MeshFunctionType, BoundaryCondition > boundaryConditionsSetter;
   boundaryConditionsSetter.template apply< typename Mesh::Cell >(
      this->boundaryCondition,
      time,
      u );

   /*cout << "u = " << u << std::endl;
  std::cout << "fu = " << fu << std::endl;
   u.save( "u.tnl" );
   fu.save( "fu.tnl" );
   getchar();*/
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
template< typename Matrix >
void
tnlMeanCurvatureFlowProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
assemblyLinearSystem( const RealType& time,
                      const RealType& tau,
                      const MeshType& mesh,
                      DofVectorType& dofsU,
                      Matrix& matrix,
                      DofVectorType& b,
                      MeshDependentDataType& meshDependentData )
{
   MeshFunctionType u( mesh, dofsU );
   tnlLinearSystemAssembler< Mesh,
			     MeshFunctionType,
			     DifferentialOperator,
			     BoundaryCondition,
			     RightHandSide,
			     tnlBackwardTimeDiscretisation,
			     MatrixType,
			     DofVectorType > systemAssembler;
   systemAssembler.template assembly< typename Mesh::Cell >(
      time,
      tau,
      mesh,
      this->differentialOperator,
      this->boundaryCondition,
      this->rightHandSide,
      u,
      matrix,
      b );
   /*matrix.print(std::cout );
  std::cout << std::endl << b << std::endl;
  std::cout << std::endl << u << std::endl;
   getchar();
   //abort();*/
}

} // namespace TNL
