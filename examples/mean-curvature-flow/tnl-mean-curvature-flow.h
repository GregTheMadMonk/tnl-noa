/***************************************************************************
                          tnl-heat-equation.h  -  description
                             -------------------
    begin                : Nov 29, 2014
    copyright            : (C) 2014 by oberhuber
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

#ifndef TNL_MEAN_CURVATIVE_FLOW_H_
#define TNL_MEAN_CURVATIVE_FLOW_H_

#include <solvers/tnlSolver.h>
#include <solvers/tnlFastBuildConfigTag.h>
#include <operators/diffusion/tnlLinearDiffusion.h>
#include <operators/tnlDirichletBoundaryConditions.h>
#include <operators/tnlNeumannBoundaryConditions.h>
#include <functions/tnlConstantFunction.h>
#include <problems/tnlMeanCurvatureFlowProblem.h>
#include <operators/diffusion/tnlNonlinearDiffusion.h>
#include <operators/operator-Q/tnlOneSideDiffOperatorQ.h>
#include <operators/operator-Q/tnlFiniteVolumeOperatorQ.h>
#include <operators/diffusion/nonlinear-diffusion-operators/tnlOneSideDiffNonlinearOperator.h>
#include <operators/diffusion/nonlinear-diffusion-operators/tnlFiniteVolumeNonlinearOperator.h>
#include <functions/tnlMeshFunction.h>

//typedef tnlDefaultConfigTag BuildConfig;
typedef tnlFastBuildConfig BuildConfig;

template< typename ConfigTag >
class meanCurvatureFlowConfig
{
   public:
      static void configSetup( tnlConfigDescription& config )
      {
         config.addDelimiter( "Mean Curvature Flow settings:" );
         config.addEntry< tnlString >( "numerical-scheme", "Numerical scheme for the solution approximation.", "fvm" );
            config.addEntryEnum< tnlString >( "fdm" );
            config.addEntryEnum< tnlString >( "fvm" );
         config.addEntry< tnlString >( "boundary-conditions-type", "Choose the boundary conditions type.", "dirichlet");
            config.addEntryEnum< tnlString >( "dirichlet" );
            config.addEntryEnum< tnlString >( "neumann" );

         config.addEntry< tnlString >( "boundary-conditions-file", "File with the values of the boundary conditions.", "boundary.tnl" );
         config.addEntry< double >( "boundary-conditions-constant", "This sets a value in case of the constant boundary conditions." );
         config.addEntry< tnlString >( "initial-condition", "File with the initial condition.", "initial.tnl");
	      config.addEntry< double >( "right-hand-side-constant", "This sets a value in case of the constant right hand side.", 0.0 );
	      config.addEntry< double >( "eps", "This sets a eps in operator Q.", 1.0 );
      };
};

template< typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag,
          typename SolverStarter >
class meanCurvatureFlowSetter
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef typename MeshType::VertexType Vertex;
   enum { Dimensions = MeshType::meshDimensions };

   static bool run( const tnlParameterContainer& parameters )
   {
      return setNumericalScheme( parameters );
   }
   
   static bool setNumericalScheme( const tnlParameterContainer& parameters )
   {
      const tnlString& numericalScheme = parameters.getParameter< tnlString >( "numerical-scheme" );
      if( numericalScheme == "fdm" )
      {
         typedef tnlOneSideDiffOperatorQ<MeshType, Real, Index > QOperator;
         typedef tnlOneSideDiffNonlinearOperator<MeshType, QOperator, Real, Index > NonlinearOperator;         
         return setBoundaryConditions< NonlinearOperator, QOperator >( parameters );
      }
      if( numericalScheme == "fvm" )
      {
         typedef tnlFiniteVolumeOperatorQ<MeshType, Real, Index, 0> QOperator;
         typedef tnlFiniteVolumeNonlinearOperator<MeshType, QOperator, Real, Index > NonlinearOperator;         
         return setBoundaryConditions< NonlinearOperator, QOperator >( parameters );
      }
      return false;
   }
   
   template< typename NonlinearOperator,
             typename QOperator >
   static bool setBoundaryConditions( const tnlParameterContainer& parameters )
   {
      typedef tnlNonlinearDiffusion< MeshType, NonlinearOperator, Real, Index > ApproximateOperator;
      typedef tnlConstantFunction< Dimensions, Real > RightHandSide;
      typedef tnlStaticVector< MeshType::meshDimensions, Real > Vertex;

      tnlString boundaryConditionsType = parameters.getParameter< tnlString >( "boundary-conditions-type" );
      if( parameters.checkParameter( "boundary-conditions-constant" ) )
      {
         typedef tnlConstantFunction< Dimensions, Real > ConstantFunction;
         if( boundaryConditionsType == "dirichlet" )
         {
            typedef tnlDirichletBoundaryConditions< MeshType, ConstantFunction, Dimensions, Real, Index > BoundaryConditions;
            typedef tnlMeanCurvatureFlowProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Solver;
            SolverStarter solverStarter;
            return solverStarter.template run< Solver >( parameters );
         }
         typedef tnlNeumannBoundaryConditions< MeshType, ConstantFunction, Real, Index > BoundaryConditions;
         typedef tnlMeanCurvatureFlowProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Solver;
         SolverStarter solverStarter;
         return solverStarter.template run< Solver >( parameters );
      }
      //typedef tnlVector< Real, Device, Index > VectorType;
      typedef tnlMeshFunction< MeshType > MeshFunction;
      if( boundaryConditionsType == "dirichlet" )
      {
         typedef tnlDirichletBoundaryConditions< MeshType, MeshFunction, Dimensions, Real, Index > BoundaryConditions;
         typedef tnlMeanCurvatureFlowProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Solver;
         SolverStarter solverStarter;
         return solverStarter.template run< Solver >( parameters );
      }
      typedef tnlNeumannBoundaryConditions< MeshType, MeshFunction, Real, Index > BoundaryConditions;
      typedef tnlMeanCurvatureFlowProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Solver;
      SolverStarter solverStarter;
      return solverStarter.template run< Solver >( parameters );
   };
};

int main( int argc, char* argv[] )
{
   tnlSolver< meanCurvatureFlowSetter, meanCurvatureFlowConfig, BuildConfig > solver;
   if( ! solver. run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

#endif /* TNL_MEAN_CURVATIVE_FLOW_H_ */
