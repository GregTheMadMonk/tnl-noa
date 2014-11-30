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
   if( ! this->boundaryCondition.setup( parameters, "boundary-conditions-" ) ||
       ! this->rightHandSide.setup( parameters, "right-hand-side-" ) )
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
   return 0;
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
                   DofVectorType& auxiliaryDofs )
{
}


template< typename Mesh, typename DifferentialOperator, typename BoundaryCondition, typename RightHandSide >
bool heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::
setInitialCondition( const tnlParameterContainer& parameters,
                     const MeshType& mesh,
                     DofVectorType& dofs,
                     DofVectorType& auxiliaryDofs )
{
   this->bindDofs( mesh, dofs );
   const tnlString& initialConditionFile = parameters.GetParameter< tnlString >( "initial-condition" );
   if( ! this->solution.load( initialConditionFile ) )
   {
      cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." << endl;
      return false;
   }
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
   const IndexType dofs = this->getDofs( mesh );
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
}

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryCondition,
          typename RightHandSide >
bool
heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::
makeSnapshot( const RealType& time,
              const IndexType& step,
              const MeshType& mesh,
              const DofVectorType& dofs,
              DofVectorType& auxiliaryDofs )
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
bool
heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::
preIterate( const RealType& time,
            const RealType& tau,
            const MeshType& mesh,
            DofVectorType& dofs,
            DofVectorType& auxDofs )
{
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

   cout << "u = " << u << endl;
   this->bindDofs( mesh, u );
   tnlExplicitUpdater< Mesh, DofVectorType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;
   explicitUpdater.template update< Mesh::Dimensions >( time,
                                                        mesh,
                                                        this->differentialOperator,
                                                        this->boundaryCondition,
                                                        this->rightHandSide,
                                                        u,
                                                        fu );
   cout << "u = " << u << endl;
   cout << "fu = " << fu << endl;
   //_u.save( "u.tnl" );
   //_fu.save( "fu.tnl" );
   //getchar();
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
   //matrix.print( cout );
   //abort();
}

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryCondition,
          typename RightHandSide >
bool
heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::
postIterate( const RealType& time,
             const RealType& tau,
             const MeshType& mesh,
             DofVectorType& dofs,
             DofVectorType& auxDofs )
{
   return true;
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
