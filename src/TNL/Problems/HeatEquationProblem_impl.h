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
#include <TNL/Logger.h>
#include <TNL/Solvers/PDE/BoundaryConditionsSetter.h>

#include "HeatEquationProblem.h"

namespace TNL {
namespace Problems {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename DifferentialOperator >
String
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, DifferentialOperator >::
getPrologHeader() const
{
   return String( "Heat equation" );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename DifferentialOperator >
void
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, DifferentialOperator >::
writeProlog( Logger& logger, const Config::ParameterContainer& parameters ) const
{
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename DifferentialOperator >
bool
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, DifferentialOperator >::
writeEpilog( Logger& logger )
{
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename DifferentialOperator >
bool
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, DifferentialOperator >::
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

   String param = parameters.getParameter< String >( "distributed-grid-io-type" );
   if( param == "MpiIO" )
        distributedIOType = Meshes::DistributedMeshes::MpiIO;
   if( param == "LocalCopy" )
        distributedIOType = Meshes::DistributedMeshes::LocalCopy;

   this->explicitUpdater.setDifferentialOperator( this->differentialOperatorPointer );
   this->explicitUpdater.setBoundaryConditions( this->boundaryConditionPointer );
   this->explicitUpdater.setRightHandSide( this->rightHandSidePointer );
   this->systemAssembler.setDifferentialOperator( this->differentialOperatorPointer );
   this->systemAssembler.setBoundaryConditions( this->boundaryConditionPointer );
   this->systemAssembler.setRightHandSide( this->rightHandSidePointer );

   this->catchExceptions = parameters.getParameter< bool >( "catch-exceptions" );
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename DifferentialOperator >
typename HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, DifferentialOperator >::IndexType
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, DifferentialOperator >::
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
          typename Communicator,
          typename DifferentialOperator >
void
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, DifferentialOperator >::
bindDofs( const DofVectorPointer& dofVector )
{
   //const IndexType dofs = this->getMesh()->template getEntitiesCount< typename MeshType::Cell >();
   this->uPointer->bind( this->getMesh(), dofVector );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename DifferentialOperator >
bool
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, DifferentialOperator >::
setInitialCondition( const Config::ParameterContainer& parameters,
                     DofVectorPointer& dofs )
{
   this->bindDofs( dofs );
   const String& initialConditionFile = parameters.getParameter< String >( "initial-condition" );
   if(CommunicatorType::isDistributed())
    {
        std::cout<<"Nodes Distribution: " << uPointer->getMesh().getDistributedMesh()->printProcessDistr() << std::endl;
        if(distributedIOType==Meshes::DistributedMeshes::MpiIO)
            Meshes::DistributedMeshes::DistributedGridIO<MeshFunctionType,Meshes::DistributedMeshes::MpiIO> ::load(initialConditionFile, *uPointer );
        if(distributedIOType==Meshes::DistributedMeshes::LocalCopy)
            Meshes::DistributedMeshes::DistributedGridIO<MeshFunctionType,Meshes::DistributedMeshes::LocalCopy> ::load(initialConditionFile, *uPointer );
        uPointer->template synchronize<CommunicatorType>();
    }
    else
    {
      if( this->catchExceptions )
      {
         try
         {
            this->uPointer->boundLoad( initialConditionFile );
         }
         catch( std::ios_base::failure& e )
         {
            std::cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." << std::endl;
            std::cerr << e.what() << std::endl;
            return false;
         }
      }
      else this->uPointer->boundLoad( initialConditionFile );
   }
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename DifferentialOperator >
   template< typename MatrixPointer >          
bool
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, DifferentialOperator >::
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
   matrixPointer->setRowCapacities( *rowLengthsPointer );
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename DifferentialOperator >
bool
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, DifferentialOperator >::
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

   if(CommunicatorType::isDistributed())
   {
      if(distributedIOType==Meshes::DistributedMeshes::MpiIO)
        Meshes::DistributedMeshes::DistributedGridIO<MeshFunctionType,Meshes::DistributedMeshes::MpiIO> ::save(fileName.getFileName(), *uPointer );
      if(distributedIOType==Meshes::DistributedMeshes::LocalCopy)
        Meshes::DistributedMeshes::DistributedGridIO<MeshFunctionType,Meshes::DistributedMeshes::LocalCopy> ::save(fileName.getFileName(), *uPointer );
   }
   else
   {
      this->uPointer->save( fileName.getFileName() );
   }
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename DifferentialOperator >
void
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, DifferentialOperator >::
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
   this->fuPointer->bind( this->getMesh(), *fuDofs );
   this->explicitUpdater.template update< typename Mesh::Cell, Communicator >( time, tau, this->getMesh(), this->uPointer, this->fuPointer );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename DifferentialOperator >
void 
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, DifferentialOperator >::
applyBoundaryConditions( const RealType& time,
                         DofVectorPointer& uDofs )
{
   this->bindDofs( uDofs );
   this->explicitUpdater.template applyBoundaryConditions< typename Mesh::Cell >( this->getMesh(), time, this->uPointer );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename DifferentialOperator >
    template< typename MatrixPointer > 
void
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, DifferentialOperator >::
assemblyLinearSystem( const RealType& time,
                      const RealType& tau,
                      const DofVectorPointer& dofsPointer,
                      MatrixPointer& matrixPointer,
                      DofVectorPointer& bPointer )
{
   this->bindDofs( dofsPointer );
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
