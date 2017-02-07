/***************************************************************************
                          HeatEquationProblem_impl.h  -  description
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

#include <TNL/FileName.h>
#include <TNL/Matrices/MatrixSetter.h>
#include <TNL/Matrices/MultidiagonalMatrixSetter.h>
#include <TNL/Logger.h>
#include <TNL/Solvers/PDE/BoundaryConditionsSetter.h>
#include <TNL/Solvers/PDE/LinearSystemAssembler.h>
#include <TNL/Solvers/PDE/BackwardTimeDiscretisation.h>

#include "HeatEquationProblem.h"

namespace TNL {
namespace Problems {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
String
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getTypeStatic()
{
   return String( "HeatEquationProblem< " ) + Mesh :: getTypeStatic() + " >";
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
String
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getPrologHeader() const
{
   return String( "Heat equation" );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
writeProlog( Logger& logger, const Config::ParameterContainer& parameters ) const
{
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
writeEpilog( Logger& logger )
{
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setup( const MeshPointer& meshPointer,
       const Config::ParameterContainer& parameters,
       const String& prefix )
{
   if( ! this->boundaryConditionPointer->setup( meshPointer, parameters, "boundary-conditions-" ) )
   {
      std::cerr << "I was not able to initialize the boundary conditions." << std::endl;
      return false;
   }
   if( ! this->rightHandSidePointer->setup( parameters, "right-hand-side-" ) )
   {
      std::cerr << "I was not able to initialize the right-hand side function." << std::endl;
      return false;
   }
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
typename HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::IndexType
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getDofs( const MeshPointer& meshPointer ) const
{
   /****
    * Set-up DOFs and supporting grid functions
    */
   return meshPointer->template getEntitiesCount< typename MeshType::Cell >();
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
bindDofs( const MeshPointer& meshPointer,
          const DofVectorPointer& dofVector )
{
   const IndexType dofs = meshPointer->template getEntitiesCount< typename MeshType::Cell >();
   this->uPointer->bind( meshPointer, dofVector );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setInitialCondition( const Config::ParameterContainer& parameters,
                     const MeshPointer& meshPointer,
                     DofVectorPointer& dofs,
                     MeshDependentDataPointer& meshDependentData )
{
   this->bindDofs( meshPointer, dofs );
   const String& initialConditionFile = parameters.getParameter< String >( "initial-condition" );
   if( ! this->uPointer->boundLoad( initialConditionFile ) )
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
   template< typename MatrixPointer >          
bool
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setupLinearSystem( const MeshPointer& meshPointer,
                   MatrixPointer& matrixPointer )
{
   const IndexType dofs = this->getDofs( meshPointer );
   typedef typename MatrixPointer::ObjectType::CompressedRowsLengthsVector CompressedRowsLengthsVectorType;
   SharedPointer< CompressedRowsLengthsVectorType > rowLengthsPointer;
   if( ! rowLengthsPointer->setSize( dofs ) )
      return false;
   Matrices::MatrixSetter< MeshType, DifferentialOperator, BoundaryCondition, CompressedRowsLengthsVectorType > matrixSetter;
   matrixSetter.template getCompressedRowsLengths< typename Mesh::Cell >(
      meshPointer,
      differentialOperatorPointer,
      boundaryConditionPointer,
      rowLengthsPointer );
   matrixPointer->setDimensions( dofs, dofs );
   if( ! matrixPointer->setCompressedRowsLengths( *rowLengthsPointer ) )
      return false;
   return true;
   //return MultidiagonalMatrixSetter< Mesh >::setupMatrix( mesh, matrix );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
makeSnapshot( const RealType& time,
              const IndexType& step,
              const MeshPointer& meshPointer,
              DofVectorPointer& dofs,
              MeshDependentDataPointer& meshDependentData )
{
   std::cout << std::endl << "Writing output at time " << time << " step " << step << "." << std::endl;

   this->bindDofs( meshPointer, dofs );

   FileName fileName;
   fileName.setFileNameBase( "u-" );
   fileName.setExtension( "tnl" );
   fileName.setIndex( step );
   if( ! this->uPointer->save( fileName.getFileName() ) )
      return false;
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getExplicitRHS( const RealType& time,
                const RealType& tau,
                const MeshPointer& meshPointer,
                DofVectorPointer& uDofs,
                DofVectorPointer& fuDofs,
                MeshDependentDataPointer& meshDependentData )
{
   /****
    * If you use an explicit solver like Euler or Merson, you
    * need to implement this method. Compute the right-hand side of
    *
    *   d/dt u(x) = fu( x, u )
    *
    * You may use supporting vectors again if you need.
    */
   
   this->bindDofs( meshPointer, uDofs );
   MeshFunctionPointer fuPointer( meshPointer, fuDofs );   
   explicitUpdater.template update< typename Mesh::Cell >(
      time,
      meshPointer,
      this->differentialOperatorPointer,
      this->boundaryConditionPointer,
      this->rightHandSidePointer,
      this->uPointer,
      fuPointer );
   //std::cerr << "******************************************************************************************" << std::endl;
   //std::cerr << "******************************************************************************************" << std::endl;
   //std::cerr << "******************************************************************************************" << std::endl;
   /*Solvers::PDE::BoundaryConditionsSetter< MeshFunctionType, BoundaryCondition > boundaryConditionsSetter;
   boundaryConditionsSetter.template apply< typename Mesh::Cell >(
      this->boundaryConditionPointer,
      time + tau,
      this->uPointer );*/
   
   /*uPointer->write( "u.txt", "gnuplot" );
   fuPointer->write( "fu.txt", "gnuplot" );
   getchar();*/
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
    template< typename MatrixPointer >          
void
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
assemblyLinearSystem( const RealType& time,
                      const RealType& tau,
                      const MeshPointer& meshPointer,
                      const DofVectorPointer& dofsPointer,
                      MatrixPointer& matrixPointer,
                      DofVectorPointer& bPointer,
                      MeshDependentDataPointer& meshDependentData )
{
   this->bindDofs( meshPointer, dofsPointer );
   Solvers::PDE::LinearSystemAssembler< Mesh,
                             MeshFunctionType,
                             DifferentialOperator,
                             BoundaryCondition,
                             RightHandSide,
                             Solvers::PDE::BackwardTimeDiscretisation,
                             typename MatrixPointer::ObjectType,
                             DofVectorType > systemAssembler;
   systemAssembler.template assembly< typename Mesh::Cell >(
      time,
      tau,
      meshPointer,
      this->differentialOperatorPointer,
      this->boundaryConditionPointer,
      this->rightHandSidePointer,
      this->uPointer,
      matrixPointer,
      bPointer );
   //matrixPointer->print( std::cout );
   //getchar();
   /*cout << endl << b << endl;
   cout << endl << u << endl;
   abort();*/
   /*cout << "Matrix multiplication test ..." << std::endl;
   Vector< RealType, DeviceType, IndexType > y;
   y.setLike( u );
   Timer timer;
   timer.reset();
   timer.start();
   for( int i = 0; i < 100; i++ )
      matrix.vectorProduct( u, y );
   timer.stop();
  std::cout << "The time is " << timer.getRealTime();
  std::cout << "Scalar product test ..." << std::endl;
   timer.reset();
   RealType a;
   timer.start();
   for( int i = 0; i < 100; i++ )
      a = y.scalarProduct( u );
   timer.stop();
  std::cout << "The time is " << timer.getRealTime();
  std::cout << std::endl;
   abort();*/
}

} // namespace Problems
} // namespace TNL
