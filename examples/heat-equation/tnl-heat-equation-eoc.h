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

#include <TNL/Solvers/Solver.h>
#include <TNL/Solvers/FastBuildConfigTag.h>
#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Functions/TestFunction.h>
#include <TNL/Operators/diffusion/LinearDiffusion.h>
#include <TNL/Operators/diffusion/ExactLinearDiffusion.h>
#include <TNL/Problems/HeatEquationEocRhs.h>
#include <TNL/Problems/HeatEquationEocProblem.h>
#include <TNL/Operators/DirichletBoundaryConditions.h>

using namespace TNL;
using namespace TNL::Problems;

//typedef tnlDefaultBuildMeshConfig BuildConfig;
typedef Solvers::FastBuildConfig BuildConfig;

template< typename MeshConfig >
class heatEquationEocConfig
{
   public:
      static void configSetup( Config::ConfigDescription& config )
      {
         config.addDelimiter( "Heat equation EOC settings:" );
         config.addDelimiter( "Tests setting::" );
         Functions::TestFunction< 3, double >::configSetup( config );
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

   typedef Containers::StaticVector< MeshType::meshDimension, Real > Point;

   static bool run( const Config::ParameterContainer& parameters )
   {
      enum { Dimension = MeshType::meshDimension };
      typedef Operators::LinearDiffusion< MeshType, Real, Index > ApproximateOperator;
      typedef Operators::ExactLinearDiffusion< Dimension > ExactOperator;
      typedef Functions::TestFunction< MeshType::meshDimension, Real, Device > TestFunction;
      typedef HeatEquationEocRhs< ExactOperator, TestFunction > RightHandSide;
      typedef Containers::StaticVector < MeshType::meshDimension, Real > Point;
      typedef Operators::DirichletBoundaryConditions< MeshType, TestFunction, Dimension, Real, Index > BoundaryConditions;
      typedef HeatEquationEocProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Solver;
      SolverStarter solverStarter;
      return solverStarter.template run< Solver >( parameters );
   };
};

int main( int argc, char* argv[] )
{
   Solvers::Solver< heatEquationSetter, heatEquationEocConfig, BuildConfig > solver;
   if( ! solver. run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

#endif /* TNL_HEAT_EQUATION_EOC_H_ */
