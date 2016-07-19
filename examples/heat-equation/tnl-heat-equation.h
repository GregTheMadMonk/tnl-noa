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

#include <solvers/tnlSolver.h>
#include <solvers/tnlFastBuildConfigTag.h>
#include <solvers/tnlBuildConfigTags.h>
#include <operators/diffusion/tnlLinearDiffusion.h>
#include <operators/tnlDirichletBoundaryConditions.h>
#include <operators/tnlNeumannBoundaryConditions.h>
#include <functions/tnlConstantFunction.h>
#include <functions/tnlMeshFunction.h>
#include <problems/tnlHeatEquationProblem.h>
#include <mesh/tnlGrid.h>

using namespace TNL;

//typedef tnlDefaultBuildMeshConfig BuildConfig;
typedef tnlFastBuildConfig BuildConfig;

template< typename MeshConfig >
class heatEquationConfig
{
   public:
      static void configSetup( tnlConfigDescription& config )
      {
         config.addDelimiter( "Heat equation settings:" );
         config.addEntry< tnlString >( "boundary-conditions-type", "Choose the boundary conditions type.", "dirichlet");
            config.addEntryEnum< tnlString >( "dirichlet" );
            config.addEntryEnum< tnlString >( "neumann" );

         typedef tnlGrid< 1, double, tnlHost, int > Mesh;
         typedef tnlMeshFunction< Mesh > MeshFunction;
         tnlDirichletBoundaryConditions< Mesh, MeshFunction >::configSetup( config );
         tnlDirichletBoundaryConditions< Mesh, tnlConstantFunction< 1 > >::configSetup( config );
         config.addEntry< tnlString >( "boundary-conditions-file", "File with the values of the boundary conditions.", "boundary.tnl" );
         config.addEntry< double >( "boundary-conditions-constant", "This sets a value in case of the constant boundary conditions." );
         config.addEntry< double >( "right-hand-side-constant", "This sets a constant value for the right-hand side.", 0.0 );
         //config.addEntry< tnlString >( "initial-condition", "File with the initial condition.", "initial.tnl");
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

   typedef tnlStaticVector< MeshType::meshDimensions, Real > Vertex;

   static bool run( const tnlParameterContainer& parameters )
   {
      enum { Dimensions = MeshType::meshDimensions };
      typedef tnlLinearDiffusion< MeshType, Real, Index > ApproximateOperator;
      typedef tnlConstantFunction< Dimensions, Real > RightHandSide;
      typedef tnlStaticVector < MeshType::meshDimensions, Real > Vertex;

      tnlString boundaryConditionsType = parameters.getParameter< tnlString >( "boundary-conditions-type" );
      if( parameters.checkParameter( "boundary-conditions-constant" ) )
      {
         typedef tnlConstantFunction< Dimensions, Real > ConstantFunction;
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
      typedef tnlMeshFunction< MeshType > MeshFunction;
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
   if( ! tnlSolver< heatEquationSetter, heatEquationConfig, BuildConfig >::run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

#endif /* TNL_HEAT_EQUATION_H_ */
