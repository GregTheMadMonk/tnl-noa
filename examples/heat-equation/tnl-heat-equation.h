/***************************************************************************
                          tnl-heat-equation.h  -  description
                             -------------------
    begin                : Nov 29, 2014
    copyright            : (C) 2014 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNL_HEAT_EQUATION_H_
#define TNL_HEAT_EQUATION_H_

#include <TNL/Solvers/Solver.h>
#include <TNL/Solvers/FastBuildConfigTag.h>
#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Operators/diffusion/LinearDiffusion.h>
#include <TNL/Operators/DirichletBoundaryConditions.h>
#include <TNL/Operators/NeumannBoundaryConditions.h>
#include <TNL/Functions/Analytic/Constant.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Problems/HeatEquationProblem.h>
#include <TNL/Meshes/Grid.h>

using namespace TNL;
using namespace TNL::Problems;

//typedef tnlDefaultBuildMeshConfig BuildConfig;
typedef Solvers::tnlFastBuildConfig BuildConfig;

template< typename MeshConfig >
class heatEquationConfig
{
   public:
      static void configSetup( Config::ConfigDescription& config )
      {
         config.addDelimiter( "Heat equation settings:" );
         config.addEntry< String >( "boundary-conditions-type", "Choose the boundary conditions type.", "dirichlet");
            config.addEntryEnum< String >( "dirichlet" );
            config.addEntryEnum< String >( "neumann" );

         typedef Meshes::Grid< 1, double, Devices::Host, int > Mesh;
         typedef Functions::MeshFunction< Mesh > MeshFunction;
         Operators::DirichletBoundaryConditions< Mesh, MeshFunction >::configSetup( config );
         Operators::DirichletBoundaryConditions< Mesh, Functions::Analytic::Constant< 1 > >::configSetup( config );
         config.addEntry< String >( "boundary-conditions-file", "File with the values of the boundary conditions.", "boundary.tnl" );
         config.addEntry< double >( "boundary-conditions-constant", "This sets a value in case of the constant boundary conditions." );
         config.addEntry< double >( "right-hand-side-constant", "This sets a constant value for the right-hand side.", 0.0 );
         //config.addEntry< String >( "initial-condition", "File with the initial condition.", "initial.tnl");
      };
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

   typedef Containers::StaticVector< MeshType::meshDimensions, Real > Vertex;

   static bool run( const Config::ParameterContainer& parameters )
   {
      enum { Dimensions = MeshType::meshDimensions };
      typedef Operators::LinearDiffusion< MeshType, Real, Index > ApproximateOperator;
      typedef Functions::Analytic::Constant< Dimensions, Real > RightHandSide;
      typedef Containers::StaticVector < MeshType::meshDimensions, Real > Vertex;

      String boundaryConditionsType = parameters.getParameter< String >( "boundary-conditions-type" );
      if( parameters.checkParameter( "boundary-conditions-constant" ) )
      {
         typedef Functions::Analytic::Constant< Dimensions, Real > Constant;
         if( boundaryConditionsType == "dirichlet" )
         {
            typedef Operators::DirichletBoundaryConditions< MeshType, Constant > BoundaryConditions;
            typedef HeatEquationProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
            SolverStarter solverStarter;
            return solverStarter.template run< Problem >( parameters );
         }
         typedef Operators::NeumannBoundaryConditions< MeshType, Constant, Real, Index > BoundaryConditions;
         typedef HeatEquationProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
         SolverStarter solverStarter;
         return solverStarter.template run< Problem >( parameters );
      }
      typedef Functions::MeshFunction< MeshType > MeshFunction;
      if( boundaryConditionsType == "dirichlet" )
      {
         typedef Operators::DirichletBoundaryConditions< MeshType, MeshFunction > BoundaryConditions;
         typedef HeatEquationProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
         SolverStarter solverStarter;
         return solverStarter.template run< Problem >( parameters );
      }
      typedef Operators::NeumannBoundaryConditions< MeshType, MeshFunction, Real, Index > BoundaryConditions;
      typedef HeatEquationProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
      SolverStarter solverStarter;
      return solverStarter.template run< Problem >( parameters );
   };
};

int main( int argc, char* argv[] )
{
   if( ! Solvers::Solver< heatEquationSetter, heatEquationConfig, BuildConfig >::run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

#endif /* TNL_HEAT_EQUATION_H_ */
