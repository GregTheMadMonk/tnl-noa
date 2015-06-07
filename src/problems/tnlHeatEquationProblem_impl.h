/***************************************************************************
                          tnlHeatEquationProblem_impl.h  -  description
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

#ifndef TNLHEATEQUATIONPROBLEM_IMPL_H_
#define TNLHEATEQUATIONPROBLEM_IMPL_H_

#include <core/mfilename.h>
#include <matrices/tnlMatrixSetter.h>
#include <matrices/tnlMultidiagonalMatrixSetter.h>
#include <core/tnlLogger.h>
#include <solvers/pde/tnlExplicitUpdater.h>
#include <solvers/pde/tnlLinearSystemAssembler.h>
#include <solvers/pde/tnlBackwardTimeDiscretisation.h>


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
setup( const tnlParameterContainer& parameters )
{
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
   return mesh.getNumberOfCells();
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
tnlHeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
bindDofs( const MeshType& mesh,
          DofVectorType& dofVector )
{
   const IndexType dofs = mesh.getNumberOfCells();
   this->solution.bind( dofVector.getData(), dofs );
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
   if( ! this->solution.load( initialConditionFile ) )
   {
      cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." << endl;
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
   typedef typename Matrix::RowLengthsVector RowLengthsVectorType;
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
tnlHeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
makeSnapshot( const RealType& time,
              const IndexType& step,
              const MeshType& mesh,
              DofVectorType& dofs,
              MeshDependentDataType& meshDependentData )
{
   cout << endl << "Writing output at time " << time << " step " << step << "." << endl;

   this->bindDofs( mesh, dofs );
   //cout << "dofs = " << dofs << endl;
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
tnlHeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getExplicitRHS( const RealType& time,
                const RealType& tau,
                const MeshType& mesh,
                DofVectorType& u,
                DofVectorType& fu,
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
    template< typename Matrix >          
void
tnlHeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
assemblyLinearSystem( const RealType& time,
                      const RealType& tau,
                      const MeshType& mesh,
                      DofVectorType& u,
                      MeshDependentDataType& meshDependentData,
                      Matrix& matrix,
                      DofVectorType& b )
{
   tnlLinearSystemAssembler< Mesh,
                             DofVectorType,
                             DifferentialOperator,
                             BoundaryCondition,
                             RightHandSide,
                             tnlBackwardTimeDiscretisation,
                             Matrix > systemAssembler;
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
   /*cout << "Matrix multiplication test ..." << endl;
   tnlVector< RealType, DeviceType, IndexType > y;
   y.setLike( u );
   tnlTimerRT timer;
   timer.reset();
   timer.start();
   for( int i = 0; i < 100; i++ )
      matrix.vectorProduct( u, y );
   timer.stop();
   cout << "The time is " << timer.getTime();
   cout << "Scalar product test ..." << endl;
   timer.reset();
   RealType a;
   timer.start();
   for( int i = 0; i < 100; i++ )
      a = y.scalarProduct( u );
   timer.stop();
   cout << "The time is " << timer.getTime();
   cout << endl;
   abort();*/
}

#endif /* TNLHEATEQUATIONPROBLEM_IMPL_H_ */
