/***************************************************************************
                          simpleProblemSolver_impl.h  -  description
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

#ifndef HEATEQUATIONSOLVER_IMPL_H_
#define HEATEQUATIONSOLVER_IMPL_H_

#include <core/mfilename.h>
#include <matrices/tnlMatrixSetter.h>
#include "heatEquationSolver.h"


template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryCondition,
          typename RightHandSide >
tnlString
heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::
getTypeStatic()
{
   return tnlString( "heatEquationSolver< " ) + Mesh :: getTypeStatic() + " >";
}

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryCondition,
          typename RightHandSide >
tnlString
heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::
getPrologHeader() const
{
   return tnlString( "Heat equation" );
}

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryCondition,
          typename RightHandSide >
void
heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::
writeProlog( tnlLogger& logger, const tnlParameterContainer& parameters ) const
{
   //logger. WriteParameter< tnlString >( "Problem name:", "problem-name", parameters );
   //logger. WriteParameter< int >( "Simple parameter:", 1 );
}

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryCondition,
          typename RightHandSide >
bool
heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::
setup( const tnlParameterContainer& parameters )
{
   if( ! boundaryCondition.setup( parameters ) ||
       ! rightHandSide.setup( parameters ) )
      return false;
   return true;
}

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryCondition,
          typename RightHandSide >
typename heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::IndexType
heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::
getDofs( const Mesh& mesh ) const
{
   /****
    * Set-up DOFs and supporting grid functions
    */
   return mesh.getNumberOfCells();
}

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryCondition,
          typename RightHandSide >
typename heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::IndexType
heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::
getAuxiliaryDofs( const Mesh& mesh ) const
{
   /****
    * Set-up DOFs and supporting grid functions which will not appear in the discrete solver
    */
}

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryCondition,
          typename RightHandSide >
void
heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::
bindDofs( const MeshType& mesh,
          DofVectorType& dofVector )
{
   const IndexType dofs = mesh.getNumberOfCells();
   this->solution.bind( dofVector.getData(), dofs );
}

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryCondition,
          typename RightHandSide >
void
heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::
bindAuxiliaryDofs( const MeshType& mesh,
                   DofVectorType& auxiliaryDofVector )
{
}


template< typename Mesh, typename DifferentialOperator, typename BoundaryCondition, typename RightHandSide >
bool heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >
:: setInitialCondition( const tnlParameterContainer& parameters,
                        const MeshType& mesh )
{
   /*const tnlString& initialConditionFile = parameters.GetParameter< tnlString >( "initial-condition" );
   if( ! this->solution.load( initialConditionFile ) )
   {
      cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." << endl;
      return false;
   }*/
   return true;
}

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryCondition,
          typename RightHandSide >
bool
heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::
setupLinearSystem( const MeshType& mesh,
                   MatrixType& matrix )
{
   RowLengthsVectorType rowLengths;
   tnlMatrixSetter< MeshType, DifferentialOperator, BoundaryCondition, RowLengthsVectorType > matrixSetter;
   matrixSetter.template getRowLengths< Mesh::Dimensions >( mesh,
                                                            differentialOperator,
                                                            boundaryCondition,
                                                            rowLengths );
   matrix.setRowLengths( rowLengths );
}

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryCondition,
          typename RightHandSide >
bool
heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::
makeSnapshot( const RealType& time,
              const IndexType& step,
              const MeshType& mesh )
{
   cout << endl << "Writing output at time " << time << " step " << step << "." << endl;

   tnlString fileName;
   FileNameBaseNumberEnding( "u-", step, 5, ".tnl", fileName );
   if( ! this->solution.save( fileName ) )
      return false;
   return true;
}

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryCondition,
          typename RightHandSide >
void
heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::
getExplicitRHS( const RealType& time,
                const RealType& tau,
                const Mesh& mesh,
                DofVectorType& _u,
                DofVectorType& _fu )
{
   /****
    * If you use an explicit solver like tnlEulerSolver or tnlMersonSolver, you
    * need to implement this method. Compute the right-hand side of
    *
    *   d/dt u(x) = fu( x, u )
    *
    * You may use supporting vectors again if you need.
    */

   this->bindDofs( mesh, _u );
   tnlExplicitUpdater< Mesh, DofVectorType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;
   explicitUpdater.template update< Mesh::Dimensions >( time,
                                                        mesh,
                                                        this->differentialOperator,
                                                        this->boundaryCondition,
                                                        this->rightHandSide,
                                                        _u,
                                                        _fu );
}

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryCondition,
          typename RightHandSide >
void
heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::
assemblyLinearSystem( const RealType& time,
                      const RealType& tau,
                      const MeshType& mesh,
                      DofVectorType& u,
                      MatrixType& matrix,
                      DofVectorType& b )
{
   tnlLinearSystemAssembler< Mesh, DofVectorType, DifferentialOperator, BoundaryCondition, RightHandSide > systemAssembler;
   systemAssembler.template assembly< Mesh::Dimensions >( time,
                                                          mesh,
                                                          differentialOperator,
                                                          boundaryConditions,
                                                          rightHandSide,
                                                          u,
                                                          matrix,
                                                          b );
}

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryCondition,
          typename RightHandSide >
tnlSolverMonitor< typename heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::RealType,
                  typename heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::IndexType >*
heatEquationSolver< Mesh,DifferentialOperator,BoundaryCondition,RightHandSide >::
getSolverMonitor()
{
   return 0;
}

#endif /* HEATEQUATION_IMPL_H_ */
