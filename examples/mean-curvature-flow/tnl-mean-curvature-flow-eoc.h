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

#ifndef TNL_MEAN_CURVATURE_FLOW_EOC_H_
#define TNL_MEAN_CURVATURE_FLOW_EOC_H_

#include <solvers/tnlSolver.h>
#include <solvers/tnlFastBuildConfig.h>
#include <solvers/tnlConfigTags.h>
#include <functions/tnlTestFunction.h>
#include <operators/tnlAnalyticDirichletBoundaryConditions.h>
#include <operators/tnlAnalyticNeumannBoundaryConditions.h>
#include "tnlMeanCurvatureFlowEocRhs.h"
#include "tnlMeanCurvatureFlowEocProblem.h"
#include "tnlExactFlowDiffusion.h"
#include "tnlMeanCurvatureFlowDiffusion.h"

//typedef tnlDefaultConfigTag BuildConfig;
typedef tnlFastBuildConfig BuildConfig;

template< typename ConfigTag >
class meanCurvatureFlowEocConfig
{
   public:
      static void configSetup( tnlConfigDescription& config )
      {
         config.addDelimiter( "Mean Curvature Flow EOC settings:" );
         config.addDelimiter( "Tests setting::" );
         tnlTestFunction< 2, double >::configSetup( config );
      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag,
          typename SolverStarter >
class meanCurvatureFlowEocSetter
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef tnlStaticVector< MeshType::Dimensions, Real > Vertex;

   static bool run( const tnlParameterContainer& parameters )
   {
      enum { Dimensions = MeshType::Dimensions };
      typedef tnlMeanCurvatureFlowDiffusion< MeshType, Real, Index > ApproximateOperator;
      typedef tnlExactFlowDiffusion< Dimensions > ExactOperator;
      typedef tnlTestFunction< MeshType::Dimensions, Real, Device > TestFunction;
      typedef tnlMeanCurvatureFlowEocRhs< ExactOperator, TestFunction > RightHandSide;
      typedef tnlStaticVector < MeshType::Dimensions, Real > Vertex;
      typedef tnlAnalyticNeumannBoundaryConditions< MeshType, TestFunction, Real, Index > BoundaryConditions;
      typedef tnlMeanCurvatureFlowEocProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Solver;
      SolverStarter solverStarter;
      return solverStarter.template run< Solver >( parameters );
   };
};

int main( int argc, char* argv[] )
{
   tnlSolver< meanCurvatureFlowEocSetter, meanCurvatureFlowEocConfig, BuildConfig > solver;
   if( ! solver. run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

#endif /* TNL_MEAN_CURVATURE_FLOW_EOC_H_ */
