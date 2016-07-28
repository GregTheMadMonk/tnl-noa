/***************************************************************************
                          tnl-heat-equation-eoc.h  -  description
                             -------------------
    begin                : Nov 29, 2014
    copyright            : (C) 2014 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNL_HEAT_EQUATION_EOC_H_
#define TNL_HEAT_EQUATION_EOC_H_

#include <TNL/solvers/tnlSolver.h>
#include <TNL/solvers/tnlFastBuildConfigTag.h>
#include <TNL/solvers/tnlBuildConfigTags.h>
#include <TNL/Functions/tnlTestFunction.h>
#include <TNL/operators/diffusion/tnlLinearDiffusion.h>
#include <TNL/operators/diffusion/tnlExactLinearDiffusion.h>
#include <TNL/problems/tnlHeatEquationEocRhs.h>
#include <TNL/problems/tnlHeatEquationEocProblem.h>
#include <TNL/operators/tnlDirichletBoundaryConditions.h>

using namespace TNL;

//typedef tnlDefaultBuildMeshConfig BuildConfig;
typedef tnlFastBuildConfig BuildConfig;

template< typename MeshConfig >
class heatEquationEocConfig
{
   public:
      static void configSetup( Config::ConfigDescription& config )
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

   typedef Vectors::StaticVector< MeshType::meshDimensions, Real > Vertex;

   static bool run( const Config::ParameterContainer& parameters )
   {
      enum { Dimensions = MeshType::meshDimensions };
      typedef tnlLinearDiffusion< MeshType, Real, Index > ApproximateOperator;
      typedef tnlExactLinearDiffusion< Dimensions > ExactOperator;
      typedef tnlTestFunction< MeshType::meshDimensions, Real, Device > TestFunction;
      typedef tnlHeatEquationEocRhs< ExactOperator, TestFunction > RightHandSide;
      typedef Vectors::StaticVector < MeshType::meshDimensions, Real > Vertex;
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
