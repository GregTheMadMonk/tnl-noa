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

#ifndef TNL_HEAT_EQUATION_H_
#define TNL_HEAT_EQUATION_H_

#include <solvers/tnlSolver.h>
#include <solvers/tnlFastBuildConfig.h>
#include <solvers/tnlConfigTags.h>
#include <operators/diffusion/tnlLinearDiffusion.h>
#include <operators/tnlAnalyticDirichletBoundaryConditions.h>
#include <operators/tnlDirichletBoundaryConditions.h>
#include <operators/tnlAnalyticNeumannBoundaryConditions.h>
#include <operators/tnlNeumannBoundaryConditions.h>
#include <functors/tnlConstantFunction.h>
#include <problems/tnlHeatEquationProblem.h>

//typedef tnlDefaultConfigTag BuildConfig;
typedef tnlFastBuildConfig BuildConfig;

template< typename ConfigTag >
class heatEquationConfig
{
   public:
      static void configSetup( tnlConfigDescription& config )
      {
         config.addDelimiter( "Heat equation settings:" );
         config.addEntry< tnlString >( "boundary-conditions-type", "Choose the boundary conditions type.", "dirichlet");
            config.addEntryEnum< tnlString >( "dirichlet" );
            config.addEntryEnum< tnlString >( "neumann" );

         config.addEntry< tnlString >( "boundary-conditions-file", "File with the values of the boundary conditions.", "boundary.tnl" );
         config.addEntry< double >( "boundary-conditions-constant", "This sets a value in case of the constant boundary conditions." );
         config.addEntry< double >( "right-hand-side-constant", "This sets a constant value for the right-hand side.", 0.0 );
         config.addEntry< tnlString >( "initial-condition", "File with the initial condition.", "initial.tnl");
      };
};

template< typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag,
          typename SolverStarter >
class heatEquationSetter
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef tnlStaticVector< MeshType::Dimensions, Real > Vertex;

   static bool run( const tnlParameterContainer& parameters )
   {
      enum { Dimensions = MeshType::Dimensions };
      typedef tnlLinearDiffusion< MeshType, Real, Index > ApproximateOperator;
      typedef tnlConstantFunction< Dimensions, Real > RightHandSide;
      typedef tnlStaticVector < MeshType::Dimensions, Real > Vertex;

      tnlString boundaryConditionsType = parameters.getParameter< tnlString >( "boundary-conditions-type" );
      if( parameters.checkParameter( "boundary-conditions-constant" ) )
      {
         typedef tnlConstantFunction< Dimensions, Real > ConstantFunction;
         if( boundaryConditionsType == "dirichlet" )
         {
            typedef tnlAnalyticDirichletBoundaryConditions< MeshType, ConstantFunction, Real, Index > BoundaryConditions;
            typedef tnlHeatEquationProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Solver;
            SolverStarter solverStarter;
            return solverStarter.template run< Solver >( parameters );
         }
         typedef tnlAnalyticNeumannBoundaryConditions< MeshType, ConstantFunction, Real, Index > BoundaryConditions;
         typedef tnlHeatEquationProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Solver;
         SolverStarter solverStarter;
         return solverStarter.template run< Solver >( parameters );
      }
      typedef tnlVector< Real, Device, Index > VectorType;
      if( boundaryConditionsType == "dirichlet" )
      {
         typedef tnlDirichletBoundaryConditions< MeshType, VectorType, Real, Index > BoundaryConditions;
         typedef tnlHeatEquationProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Solver;
         SolverStarter solverStarter;
         return solverStarter.template run< Solver >( parameters );
      }
      typedef tnlNeumannBoundaryConditions< MeshType, VectorType, Real, Index > BoundaryConditions;
      typedef tnlHeatEquationProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Solver;
      SolverStarter solverStarter;
      return solverStarter.template run< Solver >( parameters );
   };
};

int main( int argc, char* argv[] )
{
   tnlSolver< heatEquationSetter, heatEquationConfig, BuildConfig > solver;
   if( ! solver. run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

#endif /* TNL_HEAT_EQUATION_H_ */
