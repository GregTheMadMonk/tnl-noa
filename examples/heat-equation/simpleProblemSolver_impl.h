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

#ifndef SIMPLEPROBLEMSOLVER_IMPL_H_
#define SIMPLEPROBLEMSOLVER_IMPL_H_

#include <core/mfilename.h>
#include "simpleProblemSolver.h"

template< typename Mesh, typename Diffusion, typename BoundaryCondition, typename RightHandSide, 
          typename TimeFunction, typename AnalyticSpaceFunction>
tnlString simpleProblemSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction> 
::getTypeStatic()
{
   /****
    * Replace 'simpleProblemSolver' by the name of your solver.
    */
   return tnlString( "simpleProblemSolver< " ) + Mesh :: getTypeStatic() + " >";
}

template< typename Mesh, typename Diffusion, typename BoundaryCondition, typename RightHandSide,
          typename TimeFunction, typename AnalyticSpaceFunction>
tnlString simpleProblemSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction>
:: getPrologHeader() const
{
   /****
    * Replace 'Simple Problem' by the your desired title in the log table.
    */
   return tnlString( "Simple Problem" );
}

template< typename Mesh, typename Diffusion, typename BoundaryCondition, typename RightHandSide,
          typename TimeFunction, typename AnalyticSpaceFunction>
void simpleProblemSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction>
:: writeProlog( tnlLogger& logger, const tnlParameterContainer& parameters ) const
{
   /****
    * In prolog, write all input parameters which define the numerical simulation.
    * Use methods:
    *
    *    logger. writeParameters< Type >( "Label:", "name", parameters );
    *
    *  or
    *
    *    logger. writeParameter< Type >( "Label:", value );
    *
    *  See tnlLogger.h for more details.
    */

   logger. WriteParameter< tnlString >( "Problem name:", "problem-name", parameters );
   logger. WriteParameter< int >( "Simple parameter:", 1 );
}

template< typename Mesh, typename Diffusion, typename BoundaryCondition, typename RightHandSide,
          typename TimeFunction, typename AnalyticSpaceFunction>
bool simpleProblemSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction>
:: init( const tnlParameterContainer& parameters )
{
   /****
    * Set-up your solver here. It means:
    * 1. Read input parameters and model coefficients like these
    */
       
   analyticSpaceFunction.init(parameters);

   /****
    * 2. Set-up geometry of the problem domain using some mesh like tnlGrid.
    * Implement additional template specializations of the method initMesh
    * if necessary.
    */
   const tnlString& meshFile = parameters.GetParameter< tnlString >( "mesh" );
   if( ! this->mesh.load( meshFile ) )
   {
      cerr << "I am not able to load the mesh from the file " << meshFile << "." << endl;
      return false;
   }
   
   /****
    * 3. Set-up DOFs and supporting grid functions
    */
   const IndexType& dofs = this->mesh.getDofs();
   dofVector. setSize(dofs);
   dofVector2. setSize(dofs);

   /****
    * You may use tnlSharedVector if you need to split the dofVector into more
    * grid functions like the following example:
    */
   this -> u. bind( & dofVector. getData()[ 0 * dofs ], dofs );
   this -> v. bind( & dofVector2. getData()[ 0 * dofs ], dofs );

   /****
    * You may now treat u and v as usual vectors and indirectly work with this->dofVector.
    */

   return true;
}

template< typename Mesh, typename Diffusion, typename BoundaryCondition, typename RightHandSide,
          typename TimeFunction, typename AnalyticSpaceFunction>
bool simpleProblemSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction>
:: setInitialCondition( const tnlParameterContainer& parameters )
{
   /****
    * Set the initial condition here. Manipulate only this -> dofVector.
    */
   const tnlString& initialConditionFile = parameters.GetParameter< tnlString >( "initial-condition" );
   if( ! this->u.load( initialConditionFile ) )
   {
      cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." << endl;
      return false;
   }
   
   boundaryCondition.applyBoundaryConditions(mesh,u,0.0,timeFunction,analyticSpaceFunction);
   
   return true;
}

template< typename Mesh, typename Diffusion, typename BoundaryCondition, typename RightHandSide,
          typename TimeFunction, typename AnalyticSpaceFunction>
bool simpleProblemSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction> 
:: makeSnapshot( const RealType& time, const IndexType& step )
{
   /****
    * Use this method to write state of the solver to file(s).
    * All data are stored in this -> dofVector. You may use
    * supporting vectors and bind them with the dofVector as before.
    */
   cout << endl << "Writing output at time " << time << " step " << step << "." << endl;
   
   /****
    * Now write them to files.
    */
   
   analyticSolution.compute(mesh,time,v,u,timeFunction,analyticSpaceFunction);
   
   tnlString fileName;
   FileNameBaseNumberEnding( "u-", step, 5, ".tnl", fileName );
   if( ! this -> u. save( fileName ) )
      return false;
   
   FileNameBaseNumberEnding( "v-", step, 5, ".tnl", fileName );
   if( ! this -> v. save( fileName ) )
      return false;

   if(time == 1.0)
      return true;
   
   return true;
}

template< typename Mesh, typename Diffusion, typename BoundaryCondition, typename RightHandSide,
          typename TimeFunction, typename AnalyticSpaceFunction>
typename simpleProblemSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction>
:: DofVectorType& simpleProblemSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction> 
:: getDofVector()
{
   /****
    * You do not need to change this usually.
    */
   return dofVector;
}

template< typename Mesh, typename Diffusion, typename BoundaryCondition, typename RightHandSide,
          typename TimeFunction, typename AnalyticSpaceFunction>
void simpleProblemSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction> 
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
      
      /****
       *  Write the host solver here.
       */
      boundaryCondition.applyBoundaryConditions(mesh, _u, time, timeFunction, analyticSpaceFunction);
      
      diffusion.getExplicitRHS(mesh,time,tau,_u,_fu);  
      
      boundaryCondition.applyBoundaryTimeDerivation(mesh, _fu, time, timeFunction, analyticSpaceFunction);
      
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
tnlSolverMonitor< typename simpleProblemSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction>:: RealType,
                  typename simpleProblemSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction> :: IndexType >*
                  simpleProblemSolver< Mesh,Diffusion,BoundaryCondition,RightHandSide,TimeFunction,AnalyticSpaceFunction> 
::  getSolverMonitor()
{
   return 0;
}

#endif /* SIMPLEPROBLEM_IMPL_H_ */
