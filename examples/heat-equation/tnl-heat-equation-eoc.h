/***************************************************************************
                          tnl-heat-equation-eoc.h  -  description
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

#ifndef TNL_HEAT_EQUATION_EOC_H_
#define TNL_HEAT_EQUATION_EOC_H_

#include <solvers/tnlSolver.h>
#include <solvers/tnlFastBuildConfigTag.h>
#include <solvers/tnlBuildConfigTags.h>
#include <functions/tnlTestFunction.h>
#include <operators/diffusion/tnlLinearDiffusion.h>
#include <operators/diffusion/tnlExactLinearDiffusion.h>
#include <problems/tnlHeatEquationEocRhs.h>
#include <problems/tnlHeatEquationEocProblem.h>
#include <operators/tnlDirichletBoundaryConditions.h>

//typedef tnlDefaultBuildMeshConfig BuildConfig;
typedef tnlFastBuildConfig BuildConfig;

template< typename MeshConfig >
class heatEquationEocConfig
{
   public:
      static void configSetup( tnlConfigDescription& config )
      {
         config.addDelimiter( "Heat equation EOC settings:" );
         config.addDelimiter( "Tests setting::" );
         tnlTestFunction< 3, double >::configSetup( config );
      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename MeshConfig,
          typename SolverStarter >
class heatEquationSetter
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef tnlStaticVector< MeshType::meshDimensions, Real > Vertex;

   static bool run( const tnlParameterContainer& parameters )
   {
      enum { Dimensions = MeshType::meshDimensions };
      typedef tnlLinearDiffusion< MeshType, Real, Index > ApproximateOperator;
      typedef tnlExactLinearDiffusion< Dimensions > ExactOperator;
      typedef tnlTestFunction< MeshType::meshDimensions, Real, Device > TestFunction;
      typedef tnlHeatEquationEocRhs< ExactOperator, TestFunction > RightHandSide;
      typedef tnlStaticVector < MeshType::meshDimensions, Real > Vertex;
      typedef tnlDirichletBoundaryConditions< MeshType, TestFunction, Dimensions, Real, Index > BoundaryConditions;
      typedef tnlHeatEquationEocProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Solver;
      SolverStarter solverStarter;
      return solverStarter.template run< Solver >( parameters );
   };
};

int main( int argc, char* argv[] )
{
   tnlSolver< heatEquationSetter, heatEquationEocConfig, BuildConfig > solver;
   if( ! solver. run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

#endif /* TNL_HEAT_EQUATION_EOC_H_ */
