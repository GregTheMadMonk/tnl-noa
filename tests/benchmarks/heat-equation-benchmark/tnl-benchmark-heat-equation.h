#include <tnlConfig.h>
#include <solvers/tnlSolver.h>
#include <solvers/tnlBuildConfigTags.h>
#include <operators/tnlDirichletBoundaryConditions.h>
#include <operators/tnlNeumannBoundaryConditions.h>
#include <functions/tnlConstantFunction.h>
#include "HeatEquationBenchmarkProblem.h"
#include "BenchmarkLaplace.h"
#include "HeatEquationBenchmarkRhs.h"
#include "HeatEquationBenchmarkBuildConfigTag.h"

typedef HeatEquationBenchmarkBuildConfigTag BuildConfig;

/****
 * Uncoment the following (and comment the previous line) for the complete build.
 * This will include support for all floating point precisions, all indexing types
 * and more solvers. You may then choose between them from the command line.
 * The compile time may, however, take tens of minutes or even several hours,
 * esppecially if CUDA is enabled. Use this, if you want, only for the final build,
 * not in the development phase.
 */
//typedef tnlDefaultConfigTag BuildConfig;

template< typename ConfigTag >class HeatEquationBenchmarkConfig
{
   public:
      static void configSetup( tnlConfigDescription & config )
      {
         config.addDelimiter( "Heat Equation Benchmark settings:" );
         config.addEntry< tnlString >( "boundary-conditions-type", "Choose the boundary conditions type.", "dirichlet");
            config.addEntryEnum< tnlString >( "dirichlet" );
            config.addEntryEnum< tnlString >( "neumann" );
         config.addEntry< double >( "boundary-conditions-constant", "This sets a value in case of the constant boundary conditions." );

         /****
          * Add definition of your solver command line arguments.
          */

      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag,
          typename SolverStarter >
class HeatEquationBenchmarkSetter
{
   public:

      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;

      static bool run( const tnlParameterContainer & parameters )
      {
          enum { Dimensions = MeshType::getMeshDimensions() };
          typedef BenchmarkLaplace< MeshType, Real, Index > ApproximateOperator;
          typedef HeatEquationBenchmarkRhs< MeshType, Real > RightHandSide;
          typedef tnlStaticVector < MeshType::getMeshDimensions(), Real > Vertex;

         /****
          * Resolve the template arguments of your solver here.
          * The following code is for the Dirichlet and the Neumann boundary conditions.
          * Both can be constant or defined as descrete values of tnlVector.
          */
          tnlString boundaryConditionsType = parameters.getParameter< tnlString >( "boundary-conditions-type" );
          if( parameters.checkParameter( "boundary-conditions-constant" ) )
          {
             typedef tnlConstantFunction< Dimensions, Real > ConstantFunction;
             if( boundaryConditionsType == "dirichlet" )
             {
                typedef tnlDirichletBoundaryConditions< MeshType, ConstantFunction, MeshType::getMeshDimensions(), Real, Index > BoundaryConditions;
                typedef HeatEquationBenchmarkProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
                SolverStarter solverStarter;
                return solverStarter.template run< Problem >( parameters );
             }
             typedef tnlNeumannBoundaryConditions< MeshType, ConstantFunction, Real, Index > BoundaryConditions;
             typedef HeatEquationBenchmarkProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );
          }
          typedef tnlMeshFunction< MeshType > MeshFunction;
          if( boundaryConditionsType == "dirichlet" )
          {
             typedef tnlDirichletBoundaryConditions< MeshType, MeshFunction, MeshType::getMeshDimensions(), Real, Index > BoundaryConditions;
             typedef HeatEquationBenchmarkProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );
          }
          typedef tnlNeumannBoundaryConditions< MeshType, MeshFunction, Real, Index > BoundaryConditions;
          typedef HeatEquationBenchmarkProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
          SolverStarter solverStarter;
          return solverStarter.template run< Problem >( parameters );
      }

};

int main( int argc, char* argv[] )
{
   tnlSolver< HeatEquationBenchmarkSetter, HeatEquationBenchmarkConfig, BuildConfig > solver;
   if( ! solver. run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

