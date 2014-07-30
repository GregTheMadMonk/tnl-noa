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
#include "heatEquationSolver.h"


template< typename Mesh, typename Diffusion, typename BoundaryCondition, typename RightHandSide, 
          typename TimeFunction, typename AnalyticSpaceFunction>
tnlString heatEquationSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction> 
::getTypeStatic()
{
   return tnlString( "heatEquationSolver< " ) + Mesh :: getTypeStatic() + " >";
}

template< typename Mesh, typename Diffusion, typename BoundaryCondition, typename RightHandSide,
          typename TimeFunction, typename AnalyticSpaceFunction>
tnlString heatEquationSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction>
:: getPrologHeader() const
{
   return tnlString( "Heat equation" );
}

template< typename Mesh, typename Diffusion, typename BoundaryCondition, typename RightHandSide,
          typename TimeFunction, typename AnalyticSpaceFunction>
void heatEquationSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction>
:: writeProlog( tnlLogger& logger, const tnlParameterContainer& parameters ) const
{
   //logger. WriteParameter< tnlString >( "Problem name:", "problem-name", parameters );
   //logger. WriteParameter< int >( "Simple parameter:", 1 );
}

template< typename Mesh, typename Diffusion, typename BoundaryCondition, typename RightHandSide,
          typename TimeFunction, typename AnalyticSpaceFunction>
bool heatEquationSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction>
:: init( const tnlParameterContainer& parameters )
{
   analyticSpaceFunction.init(parameters);
   ifLaplaceCompare = parameters.GetParameter< IndexType >( "approximation-test" );
   if((ifLaplaceCompare != 0) && (ifLaplaceCompare != 1))
   {
      cerr << "Unknown value of laplace-convergence-test parameter. Valid values are 0 or 1. You set " << ifLaplaceCompare << ". ";
      return false;
   }
   ifSolutionCompare = parameters.GetParameter< IndexType >("eoc-test");
   if((ifSolutionCompare != 0) && (ifSolutionCompare != 1))
   {
      cerr << "Unknown value of solution-convergence-test parameter. Valid values are 0 or 1. You set " << ifSolutionCompare << ". ";
      return false;
   }
   return true;
}

template< typename Mesh,
          typename Diffusion,
          typename BoundaryCondition,
          typename RightHandSide,
          typename TimeFunction,
          typename AnalyticSpaceFunction >
typename heatEquationSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction >::IndexType 
   heatEquationSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction >::getDofs( const Mesh& mesh ) const
{
   /****
    * Set-up DOFs and supporting grid functions
    */
   return mesh.getNumberOfCells();
}

template< typename Mesh,
          typename Diffusion,
          typename BoundaryCondition,
          typename RightHandSide,
          typename TimeFunction,
          typename AnalyticSpaceFunction >
typename heatEquationSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction >::IndexType
   heatEquationSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction >::getAuxiliaryDofs( const Mesh& mesh ) const
{
   /****
    * Set-up DOFs and supporting grid functions which will not appear in the discrete solver
    */
   return 3*mesh.getNumberOfCells();
}

template< typename Mesh,
          typename Diffusion,
          typename BoundaryCondition,
          typename RightHandSide,
          typename TimeFunction,
          typename AnalyticSpaceFunction >
void
heatEquationSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction >::
bindDofs( const MeshType& mesh,
          DofVectorType& dofVector )
{
   const IndexType dofs = mesh.getNumberOfCells();
   this->numericalSolution.bind( dofVector.getData(), dofs );
}

template< typename Mesh,
          typename Diffusion,
          typename BoundaryCondition,
          typename RightHandSide,
          typename TimeFunction,
          typename AnalyticSpaceFunction >
void
heatEquationSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction >::
bindAuxiliaryDofs( const MeshType& mesh,
                   DofVectorType& auxiliaryDofVector )
{
   const IndexType dofs = mesh.getNumberOfCells();
   this->exactSolution.bind( auxiliaryDofVector.getData(), dofs );
   this->analyticLaplace.bind( &auxiliaryDofVector.getData()[ dofs ], dofs );
   this->numericalLaplace.bind( &auxiliaryDofVector.getData()[ 2*dofs ], dofs );
}


template< typename Mesh, typename Diffusion, typename BoundaryCondition, typename RightHandSide,
          typename TimeFunction, typename AnalyticSpaceFunction>
bool heatEquationSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction>
:: setInitialCondition( const tnlParameterContainer& parameters,
                        const MeshType& mesh )
{
   const tnlString& initialConditionFile = parameters.GetParameter< tnlString >( "initial-condition" );
   if( ! this->numericalSolution.load( initialConditionFile ) )
   {
      cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." << endl;
      return false;
   }
   
   //boundaryCondition.applyBoundaryConditions(mesh,numericalSolution,0.0,timeFunction,analyticSpaceFunction);
   timeFunction.applyInitTimeValues( numericalSolution);
   
   return true;
}

template< typename Mesh,
          typename Diffusion,
          typename BoundaryCondition,
          typename RightHandSide,
          typename TimeFunction,
          typename AnalyticSpaceFunction >
bool heatEquationSolver< Mesh,
                         Diffusion,
                         BoundaryCondition,
                         RightHandSide,
                         TimeFunction,
                         AnalyticSpaceFunction >::
makeSnapshot( const RealType& time,
              const IndexType& step,
              const MeshType& mesh )
{
   cout << endl << "Writing output at time " << time << " step " << step << "." << endl;

   tnlString fileName;
   FileNameBaseNumberEnding( "numericalSolution-", step, 5, ".tnl", fileName );
   if( ! this->numericalSolution.save( fileName ) )
      return false;
 
   if( ifSolutionCompare == 1)
   {
      analyticSolution.computeAnalyticSolution( mesh, time, exactSolution, timeFunction, analyticSpaceFunction );
      FileNameBaseNumberEnding( "analyticSolution-", step, 5, ".tnl", fileName );
      if( ! this->exactSolution. save( fileName ) )
         return false;
   }
   
   if(ifLaplaceCompare == 1)
   {
      analyticSolution.computeLaplace( mesh, time, analyticLaplace, timeFunction, analyticSpaceFunction );
      //diffusion.getExplicitRHS( mesh, numericalSolution, numericalLaplace );
      
      tnlString fileName;
      FileNameBaseNumberEnding( "analyticLaplace", 0, 1, ".tnl", fileName );
      if( ! this -> analyticLaplace. save( fileName ) )
         return false;
      
      FileNameBaseNumberEnding( "numericalLaplace", 0, 1, ".tnl", fileName );
      if( ! this -> numericalLaplace. save( fileName ) )
         return false;
      
      exit(0);
   }
   
   return true;
}

template< typename Mesh, typename Diffusion, typename BoundaryCondition, typename RightHandSide,
          typename TimeFunction, typename AnalyticSpaceFunction>
void heatEquationSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction> 
:: GetExplicitRHS( const RealType& time,
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
   explicitUpdater.template update< Mesh::Dimensions >( time,
                                                        tau,
                                                        mesh,
                                                        this->boundaryCondition,
                                                        this->diffusion,
                                                        _u,
                                                        _fu );
}

template< typename Mesh, typename Diffusion, typename BoundaryCondition, typename RightHandSide, 
          typename TimeFunction, typename AnalyticSpaceFunction>
tnlSolverMonitor< typename heatEquationSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction>:: RealType,
                  typename heatEquationSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction> :: IndexType >*
                  heatEquationSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction> 
::  getSolverMonitor()
{
   return 0;
}

#endif /* HEATEQUATION_IMPL_H_ */
