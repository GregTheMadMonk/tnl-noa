/***************************************************************************
                          tnl-heat-equation-eoc.cpp  -  description
                             -------------------
    begin                : Sep 7, 2014
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

#include <solvers/tnlSolver.h>
#include <solvers/tnlFastBuildConfig.h>
#include <solvers/tnlConfigTags.h>
#include <functions/tnlTestFunction.h>
#include <operators/diffusion/tnlLinearDiffusion.h>
#include <operators/diffusion/tnlExactLinearDiffusion.h>
#include <operators/tnlAnalyticDirichletBoundaryConditions.h>
#include "tnlHeatEquationEocRhs.h"
#include "heatEquationSolver.h"

//typedef tnlDefaultConfigTag BuildConfig;
typedef tnlFastBuildConfig BuildConfig;

template< typename ConfigTag >
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
      typedef tnlExactLinearDiffusion< Dimensions > ExactOperator;
      typedef tnlTestFunction< MeshType::Dimensions, Real, Device > TestFunction;
      typedef tnlHeatEquationEocRhs< ExactOperator, TestFunction > RightHandSide;
      typedef tnlStaticVector < MeshType::Dimensions, Real > Vertex;
      typedef tnlAnalyticDirichletBoundaryConditions< MeshType, TestFunction, Real, Index > BoundaryConditions;
      typedef heatEquationSolver< MeshType, ApproximateOperator, BoundaryConditions, RightHandSide > Solver;
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


