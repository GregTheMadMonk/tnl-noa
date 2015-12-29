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
#include <solvers/tnlFastBuildConfigTag.h>
#include <solvers/tnlBuildConfigTags.h>
#include <functions/tnlTestFunction.h>
#include <operators/tnlDirichletBoundaryConditions.h>
#include <operators/tnlNeumannBoundaryConditions.h>
#include <problems/tnlMeanCurvatureFlowEocRhs.h>
#include <problems/tnlMeanCurvatureFlowEocProblem.h>
#include <operators/diffusion/tnlExactNonlinearDiffusion.h>
#include <operators/diffusion/tnlNonlinearDiffusion.h>
#include <operators/operator-Q/tnlOneSideDiffOperatorQ.h>
#include <operators/operator-Q/tnlFiniteVolumeOperatorQ.h>
#include <operators/diffusion/tnlExactNonlinearDiffusion.h>
#include <operators/diffusion/nonlinear-diffusion-operators/tnlOneSideDiffNonlinearOperator.h>
#include <operators/diffusion/nonlinear-diffusion-operators/tnlFiniteVolumeNonlinearOperator.h>
#include <operators/operator-Q/tnlExactOperatorQ.h>

//typedef tnlDefaultConfigTag BuildConfig;
typedef tnlFastBuildConfig BuildConfig;

template< typename ConfigTag >
class meanCurvatureFlowEocConfig
{
   public:
      static void configSetup( tnlConfigDescription& config )
      {
         config.addDelimiter( "Mean Curvature Flow EOC settings:" );
         config.addEntry< double >( "eps", "This sets a eps in operator Q.", 1.0 );
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
class meanCurvatureFlowEocSetter
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef typename MeshType::VertexType Vertex;

   static bool run( const tnlParameterContainer& parameters )
   {
      enum { Dimensions = MeshType::meshDimensions };
      typedef tnlFiniteVolumeOperatorQ<MeshType, Real, Index, 0> OperatorQ;
      typedef tnlFiniteVolumeNonlinearOperator<MeshType, OperatorQ, Real, Index > NonlinearOperator;
      typedef tnlNonlinearDiffusion< MeshType, NonlinearOperator, Real, Index > ApproximateOperator;
      typedef tnlExactNonlinearDiffusion< tnlExactOperatorQ<Dimensions>, Dimensions > ExactOperator;
      typedef tnlTestFunction< MeshType::meshDimensions, Real, Device > TestFunction;
      typedef tnlMeanCurvatureFlowEocRhs< ExactOperator, TestFunction, Dimensions > RightHandSide;
      typedef tnlStaticVector < MeshType::meshDimensions, Real > Vertex;
      typedef tnlDirichletBoundaryConditions< MeshType, TestFunction, Real, Index > BoundaryConditions;
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
