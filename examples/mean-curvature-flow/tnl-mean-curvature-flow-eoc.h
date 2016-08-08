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

#include <TNL/Solvers/Solver.h>
#include <TNL/Solvers/FastBuildConfigTag.h>
#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Functions/TestFunction.h>
#include <TNL/operators/tnlDirichletBoundaryConditions.h>
#include <TNL/operators/tnlNeumannBoundaryConditions.h>
#include <TNL/Problems/MeanCurvatureFlowEocRhs.h>
#include <TNL/Problems/MeanCurvatureFlowEocProblem.h>
#include <TNL/operators/diffusion/tnlExactNonlinearDiffusion.h>
#include <TNL/operators/diffusion/tnlNonlinearDiffusion.h>
#include <TNL/operators/operator-Q/tnlOneSideDiffOperatorQ.h>
#include <TNL/operators/operator-Q/tnlFiniteVolumeOperatorQ.h>
#include <TNL/operators/diffusion/tnlExactNonlinearDiffusion.h>
#include <TNL/operators/diffusion/nonlinear-diffusion-operators/tnlOneSideDiffNonlinearOperator.h>
#include <TNL/operators/diffusion/nonlinear-diffusion-operators/tnlFiniteVolumeNonlinearOperator.h>
#include <TNL/operators/geometric/tnlExactGradientNorm.h>

//typedef tnlDefaultConfigTag BuildConfig;
typedef tnlFastBuildConfig BuildConfig;

template< typename ConfigTag >
class meanCurvatureFlowEocConfig
{
   public:
      static void configSetup( Config::ConfigDescription& config )
      {
         config.addDelimiter( "Mean Curvature Flow EOC settings:" );         
         config.addEntry< String >( "numerical-scheme", "Numerical scheme for the solution approximation.", "fvm" );
            config.addEntryEnum< String >( "fdm" );
            config.addEntryEnum< String >( "fvm" );

         config.addEntry< double >( "eps", "This sets a eps in operator Q.", 1.0 );
         config.addDelimiter( "Tests setting::" );         
         TestFunction< 3, double >::configSetup( config );
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

   typedef typename MeshType::VertexType Vertex;
   enum { Dimensions = MeshType::meshDimensions };

   static bool run( const Config::ParameterContainer& parameters )
   {

      typedef tnlFiniteVolumeOperatorQ<MeshType, Real, Index, 0> OperatorQ;
      typedef tnlFiniteVolumeNonlinearOperator<MeshType, OperatorQ, Real, Index > NonlinearOperator;
      typedef tnlNonlinearDiffusion< MeshType, NonlinearOperator, Real, Index > ApproximateOperator;
      typedef tnlExactNonlinearDiffusion< tnlExactGradientNorm< Dimensions >, Dimensions > ExactOperator;
      typedef TestFunction< MeshType::meshDimensions, Real, Device > TestFunction;
      typedef MeanCurvatureFlowEocRhs< ExactOperator, TestFunction, Dimensions > RightHandSide;
      typedef StaticVector < MeshType::meshDimensions, Real > Vertex;
      typedef tnlDirichletBoundaryConditions< MeshType, TestFunction, Dimensions, Real, Index > BoundaryConditions;
      typedef MeanCurvatureFlowEocProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Solver;
      SolverStarter solverStarter;
      return solverStarter.template run< Solver >( parameters );
   };
};

int main( int argc, char* argv[] )
{
   Solver< meanCurvatureFlowEocSetter, meanCurvatureFlowEocConfig, BuildConfig > solver;
   if( ! solver. run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

#endif /* TNL_MEAN_CURVATURE_FLOW_EOC_H_ */
