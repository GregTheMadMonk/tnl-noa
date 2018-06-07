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

#define MPIIO
#include <TNL/Meshes/DistributedMeshes/DistributedGridIO.h>


namespace TNL {
namespace Problems {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename CommType,
          typename DifferentialOperator >
String
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, CommType, DifferentialOperator >::
getTypeStatic()
{
   return String( "HeatEquationProblem< " ) + Mesh :: getTypeStatic() + " >";
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename CommType,
          typename DifferentialOperator >
String
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, CommType, DifferentialOperator >::
getPrologHeader() const
{
   return String( "Heat equation" );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename CommType,
          typename DifferentialOperator >
void
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, CommType, DifferentialOperator >::
writeProlog( Logger& logger, const Config::ParameterContainer& parameters ) const
{
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename CommType,
          typename DifferentialOperator >
bool
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, CommType, DifferentialOperator >::
writeEpilog( Logger& logger )
{
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename CommType,
          typename DifferentialOperator >
bool
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, CommType, DifferentialOperator >::
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
          typename CommType,
          typename DifferentialOperator >
typename HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, CommType, DifferentialOperator >::IndexType
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, CommType, DifferentialOperator >::
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
          typename CommType,
          typename DifferentialOperator >
void
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, CommType, DifferentialOperator >::
bindDofs( const MeshPointer& meshPointer,
          const DofVectorPointer& dofVector )
{
   const IndexType dofs = meshPointer->template getEntitiesCount< typename MeshType::Cell >();
   this->uPointer->bind( meshPointer, dofVector );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename CommType,
          typename DifferentialOperator >
bool
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, CommType, DifferentialOperator >::
setInitialCondition( const Config::ParameterContainer& parameters,
                     const MeshPointer& meshPointer,
                     DofVectorPointer& dofs,
                     MeshDependentDataPointer& meshDependentData )
{
   this->bindDofs( meshPointer, dofs );
   const String& initialConditionFile = parameters.getParameter< String >( "initial-condition" );
   std::cout<<"setInitialCondition" <<std::endl; 
   if(CommunicatorType::isDistributed())
    {
        std::cout<<"Nodes Distribution: " << uPointer->getMesh().getDistributedMesh()->printProcessDistr() << std::endl;
        Meshes::DistributedMeshes::DistributedGridIO<MeshFunctionType,Meshes::DistributedMeshes::MpiIO> ::load(initialConditionFile, *uPointer );
        uPointer->template synchronize<CommunicatorType>();
    }
    else
    {
       if( ! this->uPointer->boundLoad( initialConditionFile ) )
       {
          std::cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." << std::endl;
          return false;
       }
    }
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename CommType,
          typename DifferentialOperator >
   template< typename MatrixPointer >          
bool
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, CommType, DifferentialOperator >::
setupLinearSystem( const MeshPointer& meshPointer,
                   MatrixPointer& matrixPointer )
{
   const IndexType dofs = this->getDofs( meshPointer );
   typedef typename MatrixPointer::ObjectType::CompressedRowLengthsVector CompressedRowLengthsVectorType;
   SharedPointer< CompressedRowLengthsVectorType > rowLengthsPointer;
   rowLengthsPointer->setSize( dofs );
   Matrices::MatrixSetter< MeshType, DifferentialOperator, BoundaryCondition, CompressedRowLengthsVectorType > matrixSetter;
   matrixSetter.template getCompressedRowLengths< typename Mesh::Cell >(
      meshPointer,
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
          typename CommType,
          typename DifferentialOperator >
bool
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, CommType, DifferentialOperator >::
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

   if(CommunicatorType::isDistributed())
   {
      Meshes::DistributedMeshes::DistributedGridIO<MeshFunctionType,Meshes::DistributedMeshes::MpiIO> ::save(fileName.getFileName(), *uPointer );
   }
   else
   {
      if( ! this->uPointer->save( fileName.getFileName() ) )
         return false;
   }
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename CommType,
          typename DifferentialOperator >
void
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, CommType, DifferentialOperator >::
getExplicitUpdate( const RealType& time,
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
   this->explicitUpdater.setDifferentialOperator( this->differentialOperatorPointer );
   this->explicitUpdater.setBoundaryConditions( this->boundaryConditionPointer );
   this->explicitUpdater.setRightHandSide( this->rightHandSidePointer );
   this->explicitUpdater.template update< typename Mesh::Cell, CommType >( time, tau, meshPointer, this->uPointer, fuPointer );

}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename CommType,
          typename DifferentialOperator >
    template< typename MatrixPointer >          
void
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, CommType, DifferentialOperator >::
assemblyLinearSystem( const RealType& time,
                      const RealType& tau,
                      const MeshPointer& meshPointer,
                      const DofVectorPointer& dofsPointer,
                      MatrixPointer& matrixPointer,
                      DofVectorPointer& bPointer,
                      MeshDependentDataPointer& meshDependentData )
{
   this->bindDofs( meshPointer, dofsPointer );
   this->systemAssembler.setDifferentialOperator( this->differentialOperatorPointer );
   this->systemAssembler.setBoundaryConditions( this->boundaryConditionPointer );
   this->systemAssembler.setRightHandSide( this->rightHandSidePointer );
   this->systemAssembler.template assembly< typename Mesh::Cell, typename MatrixPointer::ObjectType >( 
      time,
      tau,
      meshPointer,
      this->uPointer,
      matrixPointer,
      bPointer );
}

} // namespace Problems
} // namespace TNL
