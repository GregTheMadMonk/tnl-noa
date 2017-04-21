#include <TNL/tnlConfig.h>
#include <TNL/Solvers/Solver.h>
#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Operators/DirichletBoundaryConditions.h>
#include <TNL/Operators/NeumannBoundaryConditions.h>
#include <TNL/Functions/Analytic/Constant.h>
#include "advectionProblem.h"
#include "LaxFridrichs.h"
#include "advectionRhs.h"
#include "advectionBuildConfigTag.h"

using namespace TNL;

typedef advectionBuildConfigTag BuildConfig;

/****
 * Uncoment the following (and comment the previous line) for the complete build.
 * This will include support for all floating point precisions, all indexing types
 * and more solvers. You may then choose between them from the command line.
 * The compile time may, however, take tens of minutes or even several hours,
 * esppecially if CUDA is enabled. Use this, if you want, only for the final build,
 * not in the development phase.
 */
//typedef tnlDefaultConfigTag BuildConfig;

template< typename ConfigTag >class advectionConfig
{
   public:
      static void configSetup( Config::ConfigDescription & config )
      {
         config.addDelimiter( "advection settings:" );
         config.addEntry< String >( "boundary-conditions-type", "Choose the boundary conditions type.", "dirichlet");
            config.addEntryEnum< String >( "dirichlet" );
            config.addEntryEnum< String >( "neumann" );
         config.addEntry< double >( "boundary-conditions-constant", "This sets a value in case of the constant boundary conditions." );
	 config.addEntry< double >( "artifical-viscosity", "This sets value of artifical viscosity (default 1)", 1.0);
	 config.addEntry< String >( "begin", "choose begin type", "sin");
	    config.addEntryEnum< String >( "sin");
	    config.addEntryEnum< String >( "sin_square");
	 config.addEntry< double >( "advection-speedX", "This sets value of advection speed in X direction (default 1)" , 1.0);
	 config.addEntry< double >( "advection-speedY", "This sets value of advection speed in Y direction (default 1)" , 1.0);
	 config.addEntry< String >( "move", "choose movement type", "advection");
	    config.addEntryEnum< String >( "advection");
	    config.addEntryEnum< String >( "rotation");
	 config.addEntry< int >( "dimension", "choose movement typeproblem dimension", 1);
	    config.addEntryEnum< int >( 1 );
	    config.addEntryEnum< int >( 2 );
	 config.addEntry< double >( "realSize", "Real size of scheme", 1.0);

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
class advectionSetter
{
   public:

      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;

      static bool run( const Config::ParameterContainer & parameters )
      {
          enum { Dimension = MeshType::getDimension() };
          typedef LaxFridrichs< MeshType, Real, Index > ApproximateOperator;
          typedef advectionRhs< MeshType, Real > RightHandSide;
          typedef Containers::StaticVector < MeshType::getDimension(), Real > Vertex;

         /****
          * Resolve the template arguments of your solver here.
          * The following code is for the Dirichlet and the Neumann boundary conditions.
          * Both can be constant or defined as descrete values of Vector.
          */
          String boundaryConditionsType = parameters.getParameter< String >( "boundary-conditions-type" );
          if( parameters.checkParameter( "boundary-conditions-constant" ) )
          {
             typedef Functions::Analytic::Constant< Dimension, Real > Constant;
             if( boundaryConditionsType == "dirichlet" )
             {
                typedef Operators::DirichletBoundaryConditions< MeshType, Constant, MeshType::getDimension(), Real, Index > BoundaryConditions;
                typedef advectionProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
                SolverStarter solverStarter;
                return solverStarter.template run< Problem >( parameters );
             }
             typedef Operators::NeumannBoundaryConditions< MeshType, Constant, Real, Index > BoundaryConditions;
             typedef advectionProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );
          }
          typedef Functions::MeshFunction< MeshType > MeshFunction;
          if( boundaryConditionsType == "dirichlet" )
          {
             typedef Operators::DirichletBoundaryConditions< MeshType, MeshFunction, MeshType::getDimension(), Real, Index > BoundaryConditions;
             typedef advectionProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );
          }
          typedef Operators::NeumannBoundaryConditions< MeshType, MeshFunction, Real, Index > BoundaryConditions;
          typedef advectionProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
          SolverStarter solverStarter;
          return solverStarter.template run< Problem >( parameters );
      }

};

int main( int argc, char* argv[] )
{
   Solvers::Solver< advectionSetter, advectionConfig, BuildConfig > solver;
   if( ! solver. run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

