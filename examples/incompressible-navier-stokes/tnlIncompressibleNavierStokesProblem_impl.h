/***************************************************************************
                          tnlIncompressibleNavierStokesProblem_impl.h  -  description
                             -------------------
    begin                : Mar 10, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLINCOMPRESSIBLENAVIERSTOKESPROBLEM_IMPL_H_
#define TNLINCOMPRESSIBLENAVIERSTOKESPROBLEM_IMPL_H_

#include <core/mfilename.h>
#include <matrices/tnlMatrixSetter.h>
#include <matrices/tnlMultidiagonalMatrixSetter.h>
#include <core/tnlLogger.h>
#include <solvers/pde/tnlExplicitUpdater.h>
#include <solvers/pde/tnlLinearSystemAssembler.h>



template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
tnlIncompressibleNavierStokesProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setup( const tnlParameterContainer& parameters )
{
   /*if( ! this->boundaryCondition.setup( parameters, "boundary-conditions-" ) ||
       ! this->rightHandSide.setup( parameters, "right-hand-side-" ) )
      return false;
   return true;*/
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
typename tnlIncompressibleNavierStokesProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::IndexType
tnlIncompressibleNavierStokesProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getDofs( const MeshType& mesh ) const
{
   /****
    * Set-up DOFs and supporting grid functions
    */
   return mesh.getNumberOfFaces();
}

/*template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
typename tnlIncompressibleNavierStokesProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::IndexType
tnlIncompressibleNavierStokesProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getAuxiliaryDofs( const MeshType& mesh ) const
{
   return mesh.getNumberOfCells();
}*/


template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
tnlIncompressibleNavierStokesProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
bindDofs( const MeshType& mesh,
          DofVectorType& dofVector )
{
   const IndexType uDofs = mesh.getNumberOfFaces() / 2;
   this->u[ 0 ].bind( dofVector.getData(), uDofs );
   this->u[ 1 ].bind( &dofVector.getData()[ uDofs ], uDofs );
}

/*template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
tnlIncompressibleNavierStokesProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
bindAuxiliaryDofs( const MeshType& mesh,
          DofVectorType& auxDofs )
{
   const IndexType pDofs = mesh.getNumberOfCells();
   this->p.bind( auxDofs.getData(), pDofs );
}*/

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
tnlIncompressibleNavierStokesProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setInitialCondition( const tnlParameterContainer& parameters,
                     const MeshType& mesh,
                     DofVectorType& dofs,
                     DofVectorType& auxiliaryDofs )
{
   this->bindDofs( mesh, dofs );
   this->bindAuxiliaryDofs( mesh, dofs );
   this->p.setValue( 0.0 );
   this->u[ 0 ].setValue( 0.0 );
   this->u[ 1 ].setValue( 0.0 );
   
   /*const tnlString& initialConditionFile = parameters.getParameter< tnlString >( "initial-condition" );
   if( ! this->solution.load( initialConditionFile ) )
   {
      cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." << endl;
      return false;
   }*/
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
   template< typename MatrixType >          
bool
tnlIncompressibleNavierStokesProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setupLinearSystem( const MeshType& mesh,
                   MatrixType& matrix )
{
   const IndexType dofs = this->getDofs( mesh );
   typedef typename MatrixType::RowLengthsVector RowLengthsVectorType;
   RowLengthsVectorType rowLengths;
   if( ! rowLengths.setSize( dofs ) )
      return false;
   tnlMatrixSetter< MeshType, DifferentialOperator, BoundaryCondition, RowLengthsVectorType > matrixSetter;
   matrixSetter.template getRowLengths< Mesh::Dimensions >( mesh,
                                                            differentialOperator,
                                                            boundaryCondition,
                                                            rowLengths );
   matrix.setDimensions( dofs, dofs );
   if( ! matrix.setRowLengths( rowLengths ) )
      return false;
   return true;
   //return tnlMultidiagonalMatrixSetter< Mesh >::setupMatrix( mesh, matrix );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
tnlIncompressibleNavierStokesProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
makeSnapshot( const RealType& time,
              const IndexType& step,
              const MeshType& mesh,
              DofVectorType& dofs,
              DofVectorType& auxiliaryDofs )
{
   cout << endl << "Writing output at time " << time << " step " << step << "." << endl;

   this->bindDofs( mesh, dofs );
   this->bindAuxiliaryDofs( mesh, auxiliaryDofs );
   //cout << "dofs = " << dofs << endl;
   tnlString fileName;
   FileNameBaseNumberEnding( "u-", step, 5, ".tnl", fileName );
   //if( ! this->solution.save( fileName ) )
   //   return false;
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
tnlIncompressibleNavierStokesProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getExplicitRHS( const RealType& time,
                const RealType& tau,
                const MeshType& mesh,
                DofVectorType& u,
                DofVectorType& fu )
{
   /****
    * If you use an explicit solver like tnlEulerSolver or tnlMersonSolver, you
    * need to implement this method. Compute the right-hand side of
    *
    *   d/dt u(x) = fu( x, u )
    *
    * You may use supporting vectors again if you need.
    */

   //cout << "u = " << u << endl;
   this->bindDofs( mesh, u );
   
   tnlExplicitUpdater< Mesh, DofVectorType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;
   explicitUpdater.template update< Mesh::Dimensions >( time,
                                                        mesh,
                                                        this->differentialOperator,
                                                        this->boundaryCondition,
                                                        this->rightHandSide,
                                                        u,
                                                        fu );
   //cout << "u = " << u << endl;
   //cout << "fu = " << fu << endl;
   //_u.save( "u.tnl" );
   //_fu.save( "fu.tnl" );
   //getchar();
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
    template< typename MatrixType >          
void
tnlIncompressibleNavierStokesProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
assemblyLinearSystem( const RealType& time,
                      const RealType& tau,
                      const MeshType& mesh,
                      DofVectorType& u,
                      DofVectorType& auxDofs,
                      MatrixType& matrix,
                      DofVectorType& b )
{
   tnlLinearSystemAssembler< Mesh, DofVectorType, DifferentialOperator, BoundaryCondition, RightHandSide, MatrixType > systemAssembler;
   systemAssembler.template assembly< Mesh::Dimensions >( time,
                                                          tau,
                                                          mesh,
                                                          this->differentialOperator,
                                                          this->boundaryCondition,
                                                          this->rightHandSide,
                                                          u,
                                                          matrix,
                                                          b );
   /*matrix.print( cout );
   cout << endl << b << endl;
   cout << endl << u << endl;
   abort();*/
}

#endif /* TNLINCOMPRESSIBLENAVIERSTOKESPROBLEM_IMPL_H_ */
