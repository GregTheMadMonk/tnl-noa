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
   logger. WriteParameter< tnlString >( "Problem name:", "problem-name", parameters );
   logger. WriteParameter< int >( "Simple parameter:", 1 );
}

template< typename Mesh, typename Diffusion, typename BoundaryCondition, typename RightHandSide,
          typename TimeFunction, typename AnalyticSpaceFunction>
bool heatEquationSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction>
:: init( const tnlParameterContainer& parameters )
{
   analyticSpaceFunction.init(parameters);
   ifLaplaceCompare = parameters.GetParameter< IndexType >( "laplace-convergence-test" );
   if((ifLaplaceCompare != 0) && (ifLaplaceCompare != 1))
   {
      cerr << "Unknown value of laplace-convergence-test parameter. Valid values are 0 or 1. You set " << ifLaplaceCompare << ". ";
      return false;
   }
   ifSolutionCompare = parameters.GetParameter< IndexType >("solution-convergence-test");
   if((ifSolutionCompare != 0) && (ifSolutionCompare != 1))
   {
      cerr << "Unknown value of solution-convergence-test parameter. Valid values are 0 or 1. You set " << ifSolutionCompare << ". ";
      return false;
   }
   
   const tnlString& meshFile = parameters.GetParameter< tnlString >( "mesh" );
   if( ! this->mesh.load( meshFile ) )
   {
      cerr << "I am not able to load the mesh from the file " << meshFile << "." << endl;
      return false;
   }
   
   const IndexType& dofs = this->mesh.getDofs();
   dofVectorAnalyticSolution. setSize(dofs);
   dofVectorNumericalSolution. setSize(dofs);
   analyticLaplace. setSize(dofs);
   numericalLaplace. setSize(dofs);
   
   this -> sharedVectorAnalyticSolution. bind( & dofVectorAnalyticSolution. getData()[ 0 * dofs ], dofs );
   this -> sharedVectorNumericalSolution. bind( & dofVectorNumericalSolution. getData()[ 0 * dofs ], dofs );

   return true;
}

template< typename Mesh, typename Diffusion, typename BoundaryCondition, typename RightHandSide,
          typename TimeFunction, typename AnalyticSpaceFunction>
bool heatEquationSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction>
:: setInitialCondition( const tnlParameterContainer& parameters )
{
   const tnlString& initialConditionFile = parameters.GetParameter< tnlString >( "initial-condition" );
   if( ! this->sharedVectorNumericalSolution.load( initialConditionFile ) )
   {
      cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." << endl;
      return false;
   }
   
   boundaryCondition.applyBoundaryConditions(mesh,sharedVectorNumericalSolution,0.0,timeFunction,analyticSpaceFunction);
   timeFunction.applyInitTimeValues(sharedVectorNumericalSolution);
   
   return true;
}

template< typename Mesh, typename Diffusion, typename BoundaryCondition, typename RightHandSide,
          typename TimeFunction, typename AnalyticSpaceFunction>
bool heatEquationSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction> 
:: makeSnapshot( const RealType& time, const IndexType& step )
{
   cout << endl << "Writing output at time " << time << " step " << step << "." << endl;

   tnlString fileName;
   FileNameBaseNumberEnding( "numericalSolution-", step, 5, ".tnl", fileName );
   if( ! this -> sharedVectorNumericalSolution. save( fileName ) )
      return false;
 
   if( ifSolutionCompare == 1)
   {
      analyticSolution.computeAnalyticSolution(mesh,time,sharedVectorAnalyticSolution,timeFunction,analyticSpaceFunction);
   
   FileNameBaseNumberEnding( "analyticSolution-", step, 5, ".tnl", fileName );
   if( ! this -> sharedVectorAnalyticSolution. save( fileName ) )
      return false;
   }
   
   if(ifLaplaceCompare == 1)
   {
      analyticSolution.computeLaplace(mesh, time, analyticLaplace, timeFunction, analyticSpaceFunction);
      diffusion.getExplicitRHS(mesh, dofVectorNumericalSolution, numericalLaplace);
      
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
typename heatEquationSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction>
:: DofVectorType& heatEquationSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction> 
:: getDofVector()
{
   return dofVectorNumericalSolution;
}

template< typename Mesh, typename Diffusion, typename BoundaryCondition, typename RightHandSide,
          typename TimeFunction, typename AnalyticSpaceFunction>
void heatEquationSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction> 
:: GetExplicitRHS( const RealType& time, const RealType& tau, DofVectorType& _u, DofVectorType& _fu )
{
   /****
    * If you use an explicit solver like tnlEulerSolver or tnlMersonSolver, you
    * need to implement this method. Compute the right-hand side of
    *
    *   d/dt u(x) = fu( x, u )
    *
    * You may use supporting vectors again if you need.
    */

   if( DeviceType :: getDevice() == tnlHostDevice )
   {
      boundaryCondition.applyBoundaryConditions(mesh, _u, time, timeFunction, analyticSpaceFunction);
      
      diffusion.getExplicitRHS(mesh,_u,_fu);  
      
      RHS.applyRHSValues(mesh, time, _fu, timeFunction, analyticSpaceFunction);
      
   }
#ifdef HAVE_CUDA
   if( DeviceType :: getDevice() == tnlCudaDevice )
   {
      /****
       * Write the CUDA solver here.
       */
   }
#endif
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
