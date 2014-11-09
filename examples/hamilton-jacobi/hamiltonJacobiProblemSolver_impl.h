/***************************************************************************
                          hamiltonJacobiProblemSolver_impl.h  -  description
                             -------------------
    begin                : Jul 8 , 2014
    copyright            : (C) 2014 by Tomas Sobotik
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef HAMILTONJACOBIPROBLEMSOLVER_IMPL_H_
#define HAMILTONJACOBIPROBLEMSOLVER_IMPL_H_

#include <core/mfilename.h>
#include <matrices/tnlMatrixSetter.h>

template< typename Mesh,
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
tnlString hamiltonJacobiProblemSolver< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide > :: getTypeStatic()
{
   return tnlString( "hamiltonJacobiSolver< " ) + Mesh  :: getTypeStatic() + " >";
}

template< typename Mesh,
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
tnlString hamiltonJacobiProblemSolver< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide > :: getPrologHeader() const
{
   return tnlString( "Hamilton-Jacobi" );
}

template< typename Mesh,
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
void hamiltonJacobiProblemSolver< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide > :: writeProlog( tnlLogger& logger,
                                                												 const tnlParameterContainer& parameters ) const
{
   //logger. WriteParameter< typename tnlString >( "Problem name:", "problem-name", parameters );
   //logger. WriteParameter< tnlString >( "Used scheme:", "scheme", parameters );
}

template< typename Mesh,
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
bool hamiltonJacobiProblemSolver< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide > :: setup( const tnlParameterContainer& parameters )
{
	   if( ! boundaryCondition.setup( parameters ) ||
	       ! rightHandSide.setup( parameters ) )
      return false;
   //return true;
/*
   const tnlString& problemName = parameters. GetParameter< tnlString >( "problem-name" );

   this->schemeTest = parameters. GetParameter< int >( "scheme-test" );
   this->tested = false;

   const tnlString& meshFile = parameters.GetParameter< tnlString >( "mesh" );
   if( ! this->mesh.load( meshFile ) )
   {
	   cerr << "I am not able to load the mesh from the file " << meshFile << "." << endl;
	   return false;
   }

   const IndexType& dofs = this->mesh.getDofs();
   dofVector. setSize( dofs );

   this -> u. bind( & dofVector. getData()[ 0 * dofs ], dofs );
   this -> v. bind( & dofVector. getData()[ 1 * dofs ], dofs );

*/
   return differentialOperator.init(parameters);

}


template< typename Mesh,
          typename HamiltonJacobi,
          typename BoundaryCondition,
          typename RightHandSide>
typename hamiltonJacobiProblemSolver< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >::IndexType
         hamiltonJacobiProblemSolver< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >::getDofs( const Mesh& mesh ) const
{
   /****
    * Set-up DOFs and supporting grid functions
    */
   return mesh.getNumberOfCells();
}

template< typename Mesh,
          typename HamiltonJacobi,
          typename BoundaryCondition,
          typename RightHandSide>
typename hamiltonJacobiProblemSolver< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >::IndexType
         hamiltonJacobiProblemSolver< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >::getAuxiliaryDofs( const Mesh& mesh ) const
{
   /****
    * Set-up DOFs and supporting grid functions which will not appear in the discrete solver
    */
}

template< typename Mesh,
          typename HamiltonJacobi,
          typename BoundaryCondition,
          typename RightHandSide>
void hamiltonJacobiProblemSolver< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >::
bindDofs( const MeshType& mesh,
          DofVectorType& dofVector )
{
   const IndexType dofs = mesh.getNumberOfCells();
   this->solution.bind( dofVector.getData(), dofs );
}

template< typename Mesh,
          typename HamiltonJacobi,
          typename BoundaryCondition,
          typename RightHandSide>
void hamiltonJacobiProblemSolver< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >::
bindAuxiliaryDofs( const MeshType& mesh,
                   DofVectorType& auxiliaryDofVector )
{
}

template< typename Mesh,
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
bool hamiltonJacobiProblemSolver< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide > :: setInitialCondition( const tnlParameterContainer& parameters,
        const MeshType& mesh,
        DofVectorType& dofs  )
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
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
bool hamiltonJacobiProblemSolver< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide > :: makeSnapshot( const RealType& time, const IndexType& step, const MeshType& mesh  )
{


   cout << endl << "Writing output at time " << time << " step " << step << "." << endl;

   tnlString fileName;
   FileNameBaseNumberEnding( "u-", step, 5, ".tnl", fileName );
   if( ! this -> solution.save( fileName ) )
	   return false;

   return true;
}


template< typename Mesh,
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
void hamiltonJacobiProblemSolver< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide > :: getExplicitRHS(  const RealType& time,
        																										   const RealType& tau,
        																										   const Mesh& mesh,
        																										   DofVectorType& _u,
        																										   DofVectorType& _fu  )
{

	/*
	if(!(this->schemeTest))
		scheme.GetExplicitRHS(time, tau, _u, _fu);
	else if(!(this->tested))
	{
		this->tested = true;
		DofVectorType tmp;
		if(tmp.setLike(_u))
			tmp = _u;
		scheme.GetExplicitRHS(time, tau, tmp, _u);

	}
	*/

	   this->bindDofs( mesh, _u );
	   explicitUpdater.template update< Mesh::Dimensions >( time,
	                                                        mesh,
	                                                        this->differentialOperator,
	                                                        this->boundaryCondition,
	                                                        this->rightHandSide,
	                                                        _u,
	                                                        _fu );
}

template< typename Mesh,
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
tnlSolverMonitor< typename hamiltonJacobiProblemSolver< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide > :: RealType,
                  typename hamiltonJacobiProblemSolver< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide > ::  IndexType >*
                  hamiltonJacobiProblemSolver< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >
::  getSolverMonitor()
{
   return 0;
}




template< typename Mesh,
          typename HamiltonJacobi,
          typename BoundaryCondition,
          typename RightHandSide >
bool
hamiltonJacobiProblemSolver< Mesh, HamiltonJacobi, BoundaryCondition, RightHandSide >::
preIterate( const RealType& time,
             const RealType& tau,
             const MeshType& mesh,
             DofVectorType& u )
{
   return true;
}

template< typename Mesh,
          typename HamiltonJacobi,
          typename BoundaryCondition,
          typename RightHandSide >
bool
hamiltonJacobiProblemSolver< Mesh, HamiltonJacobi, BoundaryCondition, RightHandSide >::
postIterate( const RealType& time,
             const RealType& tau,
             const MeshType& mesh,
             DofVectorType& u )
{
   return true;
}



template< typename Mesh,
          typename HamiltonJacobi,
          typename BoundaryCondition,
          typename RightHandSide >
void
hamiltonJacobiProblemSolver< Mesh, HamiltonJacobi, BoundaryCondition, RightHandSide >::
assemblyLinearSystem( const RealType& time,
                      const RealType& tau,
                      const MeshType& mesh,
                      DofVectorType& u,
                      MatrixType& matrix,
                      DofVectorType& b )
{
   tnlLinearSystemAssembler< Mesh, DofVectorType, HamiltonJacobi, BoundaryCondition, RightHandSide, MatrixType > systemAssembler;
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
          typename HamiltonJacobi,
          typename BoundaryCondition,
          typename RightHandSide >
bool
hamiltonJacobiProblemSolver< Mesh, HamiltonJacobi, BoundaryCondition, RightHandSide >::
setupLinearSystem( const MeshType& mesh,
                   MatrixType& matrix )
{
   const IndexType dofs = this->getDofs( mesh );
   RowLengthsVectorType rowLengths;
   if( ! rowLengths.setSize( dofs ) )
      return false;
   tnlMatrixSetter< MeshType, HamiltonJacobi, BoundaryCondition, RowLengthsVectorType > matrixSetter;
   matrixSetter.template getRowLengths< Mesh::Dimensions >( mesh,
                                                            differentialOperator,
                                                            boundaryCondition,
                                                            rowLengths );
   matrix.setDimensions( dofs, dofs );
   if( ! matrix.setRowLengths( rowLengths ) )
      return false;
   return true;
}




#endif /* HAMILTONJACOBIPROBLEMSOLVER_IMPL_H_ */
