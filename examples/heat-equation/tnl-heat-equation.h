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

#include <TNL/Solvers/tnlSolver.h>
#include <TNL/Solvers/tnlFastBuildConfigTag.h>
#include <TNL/Solvers/tnlBuildConfigTags.h>
#include <TNL/operators/diffusion/tnlLinearDiffusion.h>
#include <TNL/operators/tnlDirichletBoundaryConditions.h>
#include <TNL/operators/tnlNeumannBoundaryConditions.h>
#include <TNL/Functions/Analytic/ConstantFunction.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/problems/tnlHeatEquationProblem.h>
#include <TNL/mesh/tnlGrid.h>

using namespace TNL;

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

         typedef tnlGrid< 1, double, Devices::Host, int > Mesh;
         typedef Functions::MeshFunction< Mesh > MeshFunction;
         tnlDirichletBoundaryConditions< Mesh, MeshFunction >::configSetup( config );
         tnlDirichletBoundaryConditions< Mesh, Functions::Analytic::ConstantFunction< 1 > >::configSetup( config );
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

   typedef Vectors::StaticVector< MeshType::meshDimensions, Real > Vertex;

   static bool run( const Config::ParameterContainer& parameters )
   {
      enum { Dimensions = MeshType::meshDimensions };
      typedef tnlLinearDiffusion< MeshType, Real, Index > ApproximateOperator;
      typedef Functions::Analytic::ConstantFunction< Dimensions, Real > RightHandSide;
      typedef Vectors::StaticVector < MeshType::meshDimensions, Real > Vertex;

      String boundaryConditionsType = parameters.getParameter< String >( "boundary-conditions-type" );
      if( parameters.checkParameter( "boundary-conditions-constant" ) )
      {
         typedef Functions::Analytic::ConstantFunction< Dimensions, Real > ConstantFunction;
         if( boundaryConditionsType == "dirichlet" )
         {
            typedef tnlDirichletBoundaryConditions< MeshType, ConstantFunction > BoundaryConditions;
            typedef tnlHeatEquationProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
            SolverStarter solverStarter;
            return solverStarter.template run< Problem >( parameters );
         }
         typedef tnlNeumannBoundaryConditions< MeshType, ConstantFunction, Real, Index > BoundaryConditions;
         typedef tnlHeatEquationProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
         SolverStarter solverStarter;
         return solverStarter.template run< Problem >( parameters );
      }
      typedef Functions::MeshFunction< MeshType > MeshFunction;
      if( boundaryConditionsType == "dirichlet" )
      {
         typedef tnlDirichletBoundaryConditions< MeshType, MeshFunction > BoundaryConditions;
         typedef tnlHeatEquationProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
         SolverStarter solverStarter;
         return solverStarter.template run< Problem >( parameters );
      }
      typedef tnlNeumannBoundaryConditions< MeshType, MeshFunction, Real, Index > BoundaryConditions;
      typedef tnlHeatEquationProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
      SolverStarter solverStarter;
      return solverStarter.template run< Problem >( parameters );
   };
};

int main( int argc, char* argv[] )
{
   if( ! Solvers::tnlSolver< heatEquationSetter, heatEquationConfig, BuildConfig >::run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

#endif /* TNL_HEAT_EQUATION_H_ */
