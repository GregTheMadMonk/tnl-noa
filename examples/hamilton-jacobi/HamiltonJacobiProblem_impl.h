/***************************************************************************
                          HamiltonJacobiProblem_impl.h  -  description
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

#pragma once 

#include <core/mfilename.h>
#include <matrices/tnlMatrixSetter.h>
#include <exception>

template< typename Mesh,
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
tnlString HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide > :: getTypeStatic()
{
   return tnlString( "hamiltonJacobiSolver< " ) + Mesh  :: getTypeStatic() + " >";
}

template< typename Mesh,
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
tnlString HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide > :: getPrologHeader() const
{
   return tnlString( "Hamilton-Jacobi" );
}

template< typename Mesh,
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
void HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide > :: writeProlog( tnlLogger& logger,
                                                												 const tnlParameterContainer& parameters ) const
{
   //logger. WriteParameter< typename tnlString >( "Problem name:", "problem-name", parameters );
   //logger. WriteParameter< tnlString >( "Used scheme:", "scheme", parameters );
}

template< typename Mesh,
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
bool HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide > :: setup( const tnlParameterContainer& parameters )
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
typename HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >::IndexType
         HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >::getDofs( const MeshType& mesh ) const
{
   /****
    * Set-up DOFs and supporting grid functions
    */
   return mesh.template getEntitiesCount< typename MeshType::Cell >();
}

template< typename Mesh,
          typename HamiltonJacobi,
          typename BoundaryCondition,
          typename RightHandSide>
typename HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >::IndexType
         HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >::getAuxiliaryDofs( const MeshType& mesh ) const
{
   /****
    * Set-up DOFs and supporting grid functions which will not appear in the discrete solver
    */
}

template< typename Mesh,
          typename HamiltonJacobi,
          typename BoundaryCondition,
          typename RightHandSide>
void HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >::
bindDofs( const MeshType& mesh,
          DofVectorType& dofVector )
{
   const IndexType dofs = mesh.template getEntitiesCount< typename MeshType::Cell >();
   this->solution.bind( dofVector.getData(), dofs );
}


template< typename Mesh,
          typename HamiltonJacobi,
          typename BoundaryCondition,
          typename RightHandSide>
void HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >::
bindAuxiliaryDofs( const MeshType& mesh,
                   DofVectorType& auxiliaryDofVector )
{
}

template< typename Mesh,
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
bool
HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >::
setInitialCondition( const tnlParameterContainer& parameters,
                     const MeshType& mesh,
                     DofVectorType& dofs,
                     MeshDependentDataType& meshDependentData  )
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
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
bool 
HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >::
makeSnapshot( const RealType& time,
              const IndexType& step,
              const MeshType& mesh,
              DofVectorType& dofs,
              MeshDependentDataType& meshDependentData  )
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
void
HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >::
getExplicitRHS(  const RealType& time,
                 const RealType& tau,
                 const MeshType& mesh,
                 DofVectorType& _u,
                 DofVectorType& _fu,
                 MeshDependentDataType& meshDependentData  )
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

	//this->bindDofs( mesh, _u );
   MeshFunctionType u, fu;
   u.bind( mesh, _u );
   fu.bind( mesh, _fu );
   explicitUpdater.template update< typename MeshType::Cell >( time,
	                                                            mesh,
	                                                            this->differentialOperator,
	                                                            this->boundaryCondition,
	                                                            this->rightHandSide,
	                                                            u,
	                                                            fu );
}

template< typename Mesh,
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
tnlSolverMonitor< typename HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide > :: RealType,
                  typename HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide > ::  IndexType >*
                  HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >
::  getSolverMonitor()
{
   return 0;
}




template< typename Mesh,
          typename HamiltonJacobi,
          typename BoundaryCondition,
          typename RightHandSide >
bool
HamiltonJacobiProblem< Mesh, HamiltonJacobi, BoundaryCondition, RightHandSide >::
preIterate( const RealType& time,
             const RealType& tau,
             const MeshType& mesh,
             DofVectorType& u,
             MeshDependentDataType& meshDependentData )
{
   return true;
}

template< typename Mesh,
          typename HamiltonJacobi,
          typename BoundaryCondition,
          typename RightHandSide >
bool
HamiltonJacobiProblem< Mesh, HamiltonJacobi, BoundaryCondition, RightHandSide >::
postIterate( const RealType& time,
             const RealType& tau,
             const MeshType& mesh,
             DofVectorType& u,
             MeshDependentDataType& meshDependentData )
{
   return true;
}

