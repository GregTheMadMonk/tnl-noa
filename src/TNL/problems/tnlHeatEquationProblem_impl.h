/***************************************************************************
                          tnlHeatEquationProblem_impl.h  -  description
                             -------------------
    begin                : Mar 10, 2013
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

#include <TNL/core/mfilename.h>
#include <TNL/matrices/tnlMatrixSetter.h>
#include <TNL/matrices/tnlMultidiagonalMatrixSetter.h>
#include <TNL/core/tnlLogger.h>
#include <TNL/solvers/pde/tnlBoundaryConditionsSetter.h>
#include <TNL/solvers/pde/tnlExplicitUpdater.h>
#include <TNL/solvers/pde/tnlLinearSystemAssembler.h>
#include <TNL/solvers/pde/tnlBackwardTimeDiscretisation.h>

#include "tnlHeatEquationProblem.h"

namespace TNL {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
tnlString
tnlHeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getTypeStatic()
{
   return tnlString( "tnlHeatEquationProblem< " ) + Mesh :: getTypeStatic() + " >";
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
tnlString
tnlHeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getPrologHeader() const
{
   return tnlString( "Heat equation" );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
tnlHeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
writeProlog( tnlLogger& logger, const tnlParameterContainer& parameters ) const
{
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
tnlHeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
writeEpilog( tnlLogger& logger )
{
   logger.writeParameter< const char* >( "GPU transfer time:", "" );
   this->gpuTransferTimer.writeLog( logger, 1 );
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
tnlHeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setup( const tnlParameterContainer& parameters )
{
   this->gpuTransferTimer.reset();
   if( ! this->boundaryCondition.setup( parameters, "boundary-conditions-" ) ||
       ! this->rightHandSide.setup( parameters, "right-hand-side-" ) )
      return false;
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
typename tnlHeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::IndexType
tnlHeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getDofs( const MeshType& mesh ) const
{
   /****
    * Set-up DOFs and supporting grid functions
    */
   return mesh.template getEntitiesCount< typename MeshType::Cell >();
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
tnlHeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
bindDofs( const MeshType& mesh,
          const DofVectorType& dofVector )
{
   const IndexType dofs = mesh.template getEntitiesCount< typename MeshType::Cell >();
   this->u.bind( mesh, dofVector );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
tnlHeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setInitialCondition( const tnlParameterContainer& parameters,
                     const MeshType& mesh,
                     DofVectorType& dofs,
                     MeshDependentDataType& meshDependentData )
{
   this->bindDofs( mesh, dofs );
   const tnlString& initialConditionFile = parameters.getParameter< tnlString >( "initial-condition" );
   if( ! this->u.boundLoad( initialConditionFile ) )
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
tnlHeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setupLinearSystem( const MeshType& mesh,
                   Matrix& matrix )
{
   const IndexType dofs = this->getDofs( mesh );
   typedef typename Matrix::CompressedRowsLengthsVector CompressedRowsLengthsVectorType;
   CompressedRowsLengthsVectorType rowLengths;
   if( ! rowLengths.setSize( dofs ) )
      return false;
   tnlMatrixSetter< MeshType, DifferentialOperator, BoundaryCondition, CompressedRowsLengthsVectorType > matrixSetter;
   matrixSetter.template getCompressedRowsLengths< typename Mesh::Cell >(
      mesh,
      differentialOperator,
      boundaryCondition,
      rowLengths );
   matrix.setDimensions( dofs, dofs );
   if( ! matrix.setCompressedRowsLengths( rowLengths ) )
      return false;
   return true;
   //return tnlMultidiagonalMatrixSetter< Mesh >::setupMatrix( mesh, matrix );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
tnlHeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
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
   if( ! this->u.save( fileName ) )
      return false;
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
tnlHeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getExplicitRHS( const RealType& time,
                const RealType& tau,
                const MeshType& mesh,
                DofVectorType& uDofs,
		          DofVectorType& fuDofs,
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
 
   //cout << "u = " << u << std::endl;
   this->bindDofs( mesh, uDofs );
   MeshFunctionType fu( mesh, fuDofs );
   tnlExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;
   explicitUpdater.setGPUTransferTimer( this->gpuTransferTimer );
   explicitUpdater.template update< typename Mesh::Cell >(
      time,
      mesh,
      this->differentialOperator,
      this->boundaryCondition,
      this->rightHandSide,
      this->u,
      fu );
   tnlBoundaryConditionsSetter< MeshFunctionType, BoundaryCondition > boundaryConditionsSetter;
   boundaryConditionsSetter.template apply< typename Mesh::Cell >(
      this->boundaryCondition,
      time + tau,
      this->u );
 
   //fu.write( "fu.txt", "gnuplot" );
   //this->u.write( "u.txt", "gnuplot");
   //getchar();
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
tnlHeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
assemblyLinearSystem( const RealType& time,
                      const RealType& tau,
                      const MeshType& mesh,
                      const DofVectorType& dofs,
                      Matrix& matrix,
                      DofVectorType& b,
		                MeshDependentDataType& meshDependentData )
{
   this->bindDofs( mesh, dofs );
   tnlLinearSystemAssembler< Mesh,
                             MeshFunctionType,
                             DifferentialOperator,
                             BoundaryCondition,
                             RightHandSide,
                             tnlBackwardTimeDiscretisation,
                             Matrix,
                             DofVectorType > systemAssembler;
   systemAssembler.template assembly< typename Mesh::Cell >(
      time,
      tau,
      mesh,
      this->differentialOperator,
      this->boundaryCondition,
      this->rightHandSide,
      this->u,
      matrix,
      b );
   /*matrix.print(std::cout );
  std::cout << std::endl << b << std::endl;
  std::cout << std::endl << u << std::endl;
   abort();*/
   /*cout << "Matrix multiplication test ..." << std::endl;
   tnlVector< RealType, DeviceType, IndexType > y;
   y.setLike( u );
   tnlTimerRT timer;
   timer.reset();
   timer.start();
   for( int i = 0; i < 100; i++ )
      matrix.vectorProduct( u, y );
   timer.stop();
  std::cout << "The time is " << timer.getTime();
  std::cout << "Scalar product test ..." << std::endl;
   timer.reset();
   RealType a;
   timer.start();
   for( int i = 0; i < 100; i++ )
      a = y.scalarProduct( u );
   timer.stop();
  std::cout << "The time is " << timer.getTime();
  std::cout << std::endl;
   abort();*/
}

} // namespace TNL
