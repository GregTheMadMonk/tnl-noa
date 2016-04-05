#include <tnlConfig.h>
#include <solvers/tnlSolver.h>
#include <solvers/tnlBuildConfigTags.h>
#include <operators/tnlDirichletBoundaryConditions.h>
#include <operators/tnlNeumannBoundaryConditions.h>
#include <functions/tnlConstantFunction.h>
#include "eulerProblem.h"
#include "LaxFridrichs.h"
#include "tnlMyMixedBoundaryConditions.h"

#include "eulerRhs.h"
#include "eulerBuildConfigTag.h"

typedef eulerBuildConfigTag BuildConfig;

/****
 * Uncoment the following (and comment the previous line) for the complete build.
 * This will include support for all floating point precisions, all indexing types
 * and more solvers. You may then choose between them from the command line.
 * The compile time may, however, take tens of minutes or even several hours,
 * esppecially if CUDA is enabled. Use this, if you want, only for the final build,
 * not in the development phase.
 */
//typedef tnlDefaultConfigTag BuildConfig;

template< typename ConfigTag >class eulerConfig
{
   public:
      static void configSetup( tnlConfigDescription & config )
      {
         config.addDelimiter( "euler settings:" );
         config.addEntry< tnlString >( "boundary-conditions-type", "Choose the boundary conditions type.", "dirichlet");
            config.addEntryEnum< tnlString >( "dirichlet" );
            config.addEntryEnum< tnlString >( "neumann" );
            config.addEntryEnum< tnlString >( "mymixed" );
         config.addEntry< double >( "boundary-conditions-constant", "This sets a value in case of the constant boundary conditions." );
         config.addEntry< double >( "left-density", "This sets a value of left density." );
         config.addEntry< double >( "left-velocity", "This sets a value of left velocity." );
         config.addEntry< double >( "left-pressure", "This sets a value of left pressure." );
         config.addEntry< double >( "riemann-border", "This sets a position of discontinuity." );
         config.addEntry< double >( "right-density", "This sets a value of right density." );
         config.addEntry< double >( "right-velocity", "This sets a value of right velocity." );
         config.addEntry< double >( "right-pressure", "This sets a value of right pressure." );
         config.addEntry< double >( "gamma", "This sets a value of gamma constant." );

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
class eulerSetter
{
   public:

      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;

      static bool run( const tnlParameterContainer & parameters )
      {
          enum { Dimensions = MeshType::getMeshDimensions() };
          typedef LaxFridrichs< MeshType, Real, Index > ApproximateOperator;
          typedef eulerRhs< MeshType, Real > RightHandSide;
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
                typedef eulerProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
                SolverStarter solverStarter;
                return solverStarter.template run< Problem >( parameters );
             }
             typedef tnlNeumannBoundaryConditions< MeshType, ConstantFunction, Real, Index > BoundaryConditions;
             typedef eulerProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );
          }
          typedef tnlMeshFunction< MeshType > MeshFunction;
          if( boundaryConditionsType == "dirichlet" )
          {
             typedef tnlDirichletBoundaryConditions< MeshType, MeshFunction, MeshType::getMeshDimensions(), Real, Index > BoundaryConditions;
             typedef eulerProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );
          }
          if( boundaryConditionsType == "mymixed" )
          {
             typedef tnlMyMixedBoundaryConditions< MeshType, MeshFunction, MeshType::getMeshDimensions(), Real, Index > BoundaryConditions;
             typedef eulerProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );
          }
          if( boundaryConditionsType == "neumann" )
          {
             typedef tnlNeumannBoundaryConditions< MeshType, MeshFunction, Real, Index > BoundaryConditions;
             typedef eulerProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );
          }
        }

};

int main( int argc, char* argv[] )
{
   tnlSolver< eulerSetter, eulerConfig, BuildConfig > solver;
   if( ! solver. run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

