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

#include "HeatEquationProblem.h"

namespace TNL {
namespace Problems {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
String
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getType()
{
   return String( "HeatEquationProblem< " ) + Mesh :: getType() + " >";
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
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   if( ! this->boundaryConditionPointer->setup( this->getMesh(), parameters, "boundary-conditions-" ) )
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
getDofs() const
{
   /****
    * Set-up DOFs and supporting grid functions
    */
   return this->getMesh()->template getEntitiesCount< typename MeshType::Cell >();
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
bindDofs( const DofVectorPointer& dofVector )
{
   const IndexType dofs = this->getMesh()->template getEntitiesCount< typename MeshType::Cell >();
   this->uPointer->bind( this->getMesh(), dofVector );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setInitialCondition( const Config::ParameterContainer& parameters,
                     DofVectorPointer& dofs )
{
   this->bindDofs( dofs );
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
setupLinearSystem( MatrixPointer& matrixPointer )
{
   const IndexType dofs = this->getDofs();
   typedef typename MatrixPointer::ObjectType::CompressedRowLengthsVector CompressedRowLengthsVectorType;
   Pointers::SharedPointer<  CompressedRowLengthsVectorType > rowLengthsPointer;
   rowLengthsPointer->setSize( dofs );
   Matrices::MatrixSetter< MeshType, DifferentialOperator, BoundaryCondition, CompressedRowLengthsVectorType > matrixSetter;
   matrixSetter.template getCompressedRowLengths< typename Mesh::Cell >(
      this->getMesh(),
      differentialOperatorPointer,
      boundaryConditionPointer,
      rowLengthsPointer );
   matrixPointer->setDimensions( dofs, dofs );
   matrixPointer->setCompressedRowLengths( *rowLengthsPointer );
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
              DofVectorPointer& dofs )
{
   std::cout << std::endl << "Writing output at time " << time << " step " << step << "." << std::endl;

   this->bindDofs( dofs );

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
getExplicitUpdate( const RealType& time,
                   const RealType& tau,
                   DofVectorPointer& uDofs,
                   DofVectorPointer& fuDofs )
{
   /****
    * If you use an explicit solver like Euler or Merson, you
    * need to implement this method. Compute the right-hand side of
    *
    *   d/dt u(x) = fu( x, u )
    *
    * You may use supporting vectors again if you need.
    */
   
   this->bindDofs( uDofs );
   MeshFunctionPointer fuPointer( this->getMesh(), fuDofs );
   this->explicitUpdater.setDifferentialOperator( this->differentialOperatorPointer ),
   this->explicitUpdater.setBoundaryConditions( this->boundaryConditionPointer ),
   this->explicitUpdater.setRightHandSide( this->rightHandSidePointer ),
   this->explicitUpdater.template update< typename Mesh::Cell >( time, tau, this->getMesh(), this->uPointer, fuPointer );
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
                      const DofVectorPointer& dofsPointer,
                      MatrixPointer& matrixPointer,
                      DofVectorPointer& bPointer )
{
   this->bindDofs( dofsPointer );
   this->systemAssembler.setDifferentialOperator( this->differentialOperatorPointer );
   this->systemAssembler.setBoundaryConditions( this->boundaryConditionPointer );
   this->systemAssembler.setRightHandSide( this->rightHandSidePointer );
   this->systemAssembler.template assembly< typename Mesh::Cell, typename MatrixPointer::ObjectType >( 
      time,
      tau,
      this->getMesh(),
      this->uPointer,
      matrixPointer,
      bPointer );
}

} // namespace Problems
} // namespace TNL
