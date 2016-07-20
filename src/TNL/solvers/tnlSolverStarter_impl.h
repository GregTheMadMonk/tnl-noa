/***************************************************************************
                          tnlSolverStarter_impl.h  -  description
                             -------------------
    begin                : Mar 9, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/tnlConfig.h>
#include <TNL/Logger.h>
#include <TNL/String.h>
#include <TNL/core/tnlCuda.h>
#include <TNL/solvers/ode/tnlMersonSolver.h>
#include <TNL/solvers/ode/tnlEulerSolver.h>
#include <TNL/solvers/linear/stationary/tnlSORSolver.h>
#include <TNL/solvers/linear/krylov/tnlCGSolver.h>
#include <TNL/solvers/linear/krylov/tnlBICGStabSolver.h>
#include <TNL/solvers/linear/krylov/tnlGMRESSolver.h>
#include <TNL/solvers/linear/krylov/tnlTFQMRSolver.h>
#include <TNL/solvers/linear/tnlUmfpackWrapper.h>
#include <TNL/solvers/preconditioners/tnlDummyPreconditioner.h>
#include <TNL/solvers/preconditioners/tnlDiagonalPreconditioner.h>
#include <TNL/solvers/pde/tnlExplicitTimeStepper.h>
#include <TNL/solvers/pde/tnlSemiImplicitTimeStepper.h>
#include <TNL/solvers/pde/tnlPDESolver.h>
#include <TNL/solvers/tnlIterativeSolverMonitor.h>
#include <TNL/solvers/ode/tnlODESolverMonitor.h>

namespace TNL {

template< typename Problem,
          typename ConfigTag,
          typename TimeStepper = typename Problem::TimeStepper >
class tnlUserDefinedTimeDiscretisationSetter;

template< typename Problem,
          typename TimeDiscretisation,
          typename ConfigTag,
          bool enabled = tnlConfigTagTimeDiscretisation< ConfigTag, TimeDiscretisation >::enabled >
class tnlSolverStarterTimeDiscretisationSetter{};

template< typename Problem,
          typename ExplicitSolver,
          typename ConfigTag,
          bool enabled = tnlConfigTagExplicitSolver< ConfigTag, ExplicitSolver >::enabled >
class tnlSolverStarterExplicitSolverSetter{};

template< typename Problem,
          typename SemiImplicitSolver,
          template<typename, typename, typename> class Preconditioner,
          typename ConfigTag,
          bool enabled = tnlConfigTagSemiImplicitSolver< ConfigTag, SemiImplicitSolver >::enabled >
class tnlSolverStarterSemiImplicitSolverSetter{};


template< typename Problem,
          typename SemiImplicitSolverTag,
          typename ConfigTag >
class tnlSolverStarterPreconditionerSetter;

template< typename Problem,
          typename ExplicitSolver,
          typename TimeStepper,
          typename ConfigTag >
class tnlSolverStarterExplicitTimeStepperSetter;

template< typename Problem,
          typename TimeStepper,
          typename ConfigTag >
class tnlSolverStarterSemiImplicitTimeStepperSetter;

template< typename ConfigTag >
tnlSolverStarter< ConfigTag > :: tnlSolverStarter()
: logWidth( 80 )
{
}

template< typename ConfigTag >
   template< typename Problem >
bool tnlSolverStarter< ConfigTag > :: run( const Config::ParameterContainer& parameters )
{
   /****
    * Create and set-up the problem
    */
   if( ! tnlHost::setup( parameters ) ||
       ! tnlCuda::setup( parameters ) )
      return false;
   Problem problem;
   if( ! problem.setup( parameters ) )
   {
      std::cerr << "The problem initiation failed!" << std::endl;
      return false;
   }

   return tnlUserDefinedTimeDiscretisationSetter< Problem, ConfigTag >::run( problem, parameters );
}

template< typename Problem,
          typename ConfigTag,
          typename TimeStepper >
class tnlUserDefinedTimeDiscretisationSetter
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         TimeStepper timeStepper;
         if( ! timeStepper.setup( parameters ) )
         {
            std::cerr << "The time stepper initiation failed!" << std::endl;
            return false;
         }
         tnlSolverStarter< ConfigTag > solverStarter;
         return solverStarter.template runPDESolver< Problem, TimeStepper >( problem, parameters, timeStepper );
      }
};

template< typename Problem,
          typename ConfigTag >
class tnlUserDefinedTimeDiscretisationSetter< Problem, ConfigTag, void >
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         /****
          * Set-up the time discretisation
          */
         const String& timeDiscretisation = parameters. getParameter< String>( "time-discretisation" );
         if( timeDiscretisation == "explicit" )
            return tnlSolverStarterTimeDiscretisationSetter< Problem, tnlExplicitTimeDiscretisationTag, ConfigTag >::run( problem, parameters );
         if( timeDiscretisation == "semi-implicit" )
            return tnlSolverStarterTimeDiscretisationSetter< Problem, tnlSemiImplicitTimeDiscretisationTag, ConfigTag >::run( problem, parameters );
         if( timeDiscretisation == "implicit" )
            return tnlSolverStarterTimeDiscretisationSetter< Problem, tnlImplicitTimeDiscretisationTag, ConfigTag >::run( problem, parameters );
         std::cerr << "Uknown time discretisation: " << timeDiscretisation << "." << std::endl;
         return false;
      }
};

/****
 * Setting the time discretisation
 */

template< typename Problem,
          typename TimeDiscretisation,
          typename ConfigTag >
class tnlSolverStarterTimeDiscretisationSetter< Problem, TimeDiscretisation, ConfigTag, false >
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         std::cerr << "The time discretisation " << parameters.getParameter< String >( "time-discretisation" ) << " is not supported." << std::endl;
         return false;
      }
};

template< typename Problem,
          typename ConfigTag >
class tnlSolverStarterTimeDiscretisationSetter< Problem, tnlExplicitTimeDiscretisationTag, ConfigTag, true >
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         const String& discreteSolver = parameters. getParameter< String>( "discrete-solver" );
         if( discreteSolver != "euler" &&
             discreteSolver != "merson" )
         {
            std::cerr << "Unknown explicit discrete solver " << discreteSolver << ". It can be only: euler or merson." << std::endl;
            return false;
         }
         if( discreteSolver == "euler" )
            return tnlSolverStarterExplicitSolverSetter< Problem, tnlExplicitEulerSolverTag, ConfigTag >::run( problem, parameters );
         if( discreteSolver == "merson" )
            return tnlSolverStarterExplicitSolverSetter< Problem, tnlExplicitMersonSolverTag, ConfigTag >::run( problem, parameters );
         return false;
      }
};

template< typename Problem,
          typename ConfigTag >
class tnlSolverStarterTimeDiscretisationSetter< Problem, tnlSemiImplicitTimeDiscretisationTag, ConfigTag, true >
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         const String& discreteSolver = parameters. getParameter< String>( "discrete-solver" );
#ifndef HAVE_UMFPACK
         if( discreteSolver != "sor" &&
             discreteSolver != "cg" &&
             discreteSolver != "bicgstab" &&
             discreteSolver != "gmres" &&
             discreteSolver != "tfqmr" )
         {
            std::cerr << "Unknown semi-implicit discrete solver " << discreteSolver << ". It can be only: sor, cg, bicgstab, gmres or tfqmr." << std::endl;
            return false;
         }
#else
         if( discreteSolver != "sor" &&
             discreteSolver != "cg" &&
             discreteSolver != "bicgstab" &&
             discreteSolver != "gmres" &&
             discreteSolver != "tfqmr" &&
             discreteSolver != "umfpack" )
         {
            std::cerr << "Unknown semi-implicit discrete solver " << discreteSolver << ". It can be only: sor, cg, bicgstab, gmres, tfqmr or umfpack." << std::endl;
            return false;
         }
#endif

         if( discreteSolver == "sor" )
            return tnlSolverStarterPreconditionerSetter< Problem, tnlSemiImplicitSORSolverTag, ConfigTag >::run( problem, parameters );
         if( discreteSolver == "cg" )
            return tnlSolverStarterPreconditionerSetter< Problem, tnlSemiImplicitCGSolverTag, ConfigTag >::run( problem, parameters );
         if( discreteSolver == "bicgstab" )
            return tnlSolverStarterPreconditionerSetter< Problem, tnlSemiImplicitBICGStabSolverTag, ConfigTag >::run( problem, parameters );
         if( discreteSolver == "gmres" )
            return tnlSolverStarterPreconditionerSetter< Problem, tnlSemiImplicitGMRESSolverTag, ConfigTag >::run( problem, parameters );
         if( discreteSolver == "tfqmr" )
            return tnlSolverStarterPreconditionerSetter< Problem, tnlSemiImplicitTFQMRSolverTag, ConfigTag >::run( problem, parameters );
#ifdef HAVE_UMFPACK
         if( discreteSolver == "umfpack" )
            return tnlSolverStarterPreconditionerSetter< Problem, tnlSemiImplicitUmfpackSolverTag, ConfigTag >::run( problem, parameters );
#endif
         return false;
      }
};

template< typename Problem,
          typename ConfigTag >
class tnlSolverStarterTimeDiscretisationSetter< Problem, tnlImplicitTimeDiscretisationTag, ConfigTag, true >
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         const String& discreteSolver = parameters. getParameter< String>( "discrete-solver" );
         return false;
      }
};

/****
 * Setting the explicit solver
 */

template< typename Problem,
          typename ExplicitSolverTag,
          typename ConfigTag >
class tnlSolverStarterExplicitSolverSetter< Problem, ExplicitSolverTag, ConfigTag, false >
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         std::cerr << "The explicit solver " << parameters.getParameter< String >( "discrete-solver" ) << " is not supported." << std::endl;
         return false;
      }
};

template< typename Problem,
          typename ExplicitSolverTag,
          typename ConfigTag >
class tnlSolverStarterExplicitSolverSetter< Problem, ExplicitSolverTag, ConfigTag, true >
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         typedef tnlExplicitTimeStepper< Problem, ExplicitSolverTag::template Template > TimeStepper;
         typedef typename ExplicitSolverTag::template Template< TimeStepper > ExplicitSolver;
         return tnlSolverStarterExplicitTimeStepperSetter< Problem,
                                                           ExplicitSolver,
                                                           TimeStepper,
                                                           ConfigTag >::run( problem, parameters );
      }
};

/****
 * Setting the semi-implicit solver
 */

template< typename Problem,
          typename SemiImplicitSolverTag,
          typename ConfigTag >
class tnlSolverStarterPreconditionerSetter
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         const String& preconditioner = parameters.getParameter< String>( "preconditioner" );

         if( preconditioner == "none" )
            return tnlSolverStarterSemiImplicitSolverSetter< Problem, SemiImplicitSolverTag, tnlDummyPreconditioner, ConfigTag >::run( problem, parameters );
         if( preconditioner == "diagonal" )
            return tnlSolverStarterSemiImplicitSolverSetter< Problem, SemiImplicitSolverTag, tnlDiagonalPreconditioner, ConfigTag >::run( problem, parameters );

         std::cerr << "Unknown preconditioner " << preconditioner << ". It can be only: none, diagonal." << std::endl;
         return false;
      }
};

template< typename Problem,
          typename SemiImplicitSolverTag,
          template<typename, typename, typename> class Preconditioner,
          typename ConfigTag >
class tnlSolverStarterSemiImplicitSolverSetter< Problem, SemiImplicitSolverTag, Preconditioner, ConfigTag, false >
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         std::cerr << "The semi-implicit solver " << parameters.getParameter< String >( "discrete-solver" ) << " is not supported." << std::endl;
         return false;
      }
};

template< typename Problem,
          typename SemiImplicitSolverTag,
          template<typename, typename, typename> class Preconditioner,
          typename ConfigTag >
class tnlSolverStarterSemiImplicitSolverSetter< Problem, SemiImplicitSolverTag, Preconditioner, ConfigTag, true >
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         typedef typename Problem::MatrixType MatrixType;
         typedef typename MatrixType::RealType RealType;
         typedef typename MatrixType::DeviceType DeviceType;
         typedef typename MatrixType::IndexType IndexType;
         typedef typename SemiImplicitSolverTag::template Template< MatrixType, Preconditioner< RealType, DeviceType, IndexType > > LinearSystemSolver;
         typedef tnlSemiImplicitTimeStepper< Problem, LinearSystemSolver > TimeStepper;
         return tnlSolverStarterSemiImplicitTimeStepperSetter< Problem,
                                                               TimeStepper,
                                                               ConfigTag >::run( problem, parameters );
      }
};

/****
 * Setting the explicit time stepper
 */

template< typename Problem,
          typename ExplicitSolver,
          typename TimeStepper,
          typename ConfigTag >
class tnlSolverStarterExplicitTimeStepperSetter
{
   public:

      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters)
      {
         typedef typename Problem::RealType RealType;
         typedef typename Problem::IndexType IndexType;
         typedef tnlODESolverMonitor< RealType, IndexType > SolverMonitorType;

         const int verbose = parameters.getParameter< int >( "verbose" );

         ExplicitSolver explicitSolver;
         explicitSolver.setup( parameters );
         explicitSolver.setVerbose( verbose );

         SolverMonitorType odeSolverMonitor;
         odeSolverMonitor.setVerbose( verbose );
         if( ! problem.getSolverMonitor() )
            explicitSolver.setSolverMonitor( odeSolverMonitor );
         else
            explicitSolver.setSolverMonitor( * ( SolverMonitorType* ) problem. getSolverMonitor() );

         TimeStepper timeStepper;
         if( ! timeStepper.setup( parameters ) )
         {
            std::cerr << "The time stepper initiation failed!" << std::endl;
            return false;
         }
         timeStepper.setSolver( explicitSolver );

         tnlSolverStarter< ConfigTag > solverStarter;
         return solverStarter.template runPDESolver< Problem, TimeStepper >( problem, parameters, timeStepper );
      };
};

/****
 * Setting the semi-implicit time stepper
 */
template< typename Problem,
          typename TimeStepper,
          typename ConfigTag >
class tnlSolverStarterSemiImplicitTimeStepperSetter
{
   public:

      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters)
      {
         typedef typename TimeStepper::LinearSystemSolverType LinearSystemSolverType;
         typedef typename LinearSystemSolverType::PreconditionerType PreconditionerType;
         typedef typename LinearSystemSolverType::MatrixType MatrixType;
         typedef typename Problem::RealType RealType;
         typedef typename Problem::IndexType IndexType;
         typedef tnlIterativeSolverMonitor< RealType, IndexType > SolverMonitorType;

         const int verbose = parameters.getParameter< int >( "verbose" );

         LinearSystemSolverType linearSystemSolver;
         linearSystemSolver.setup( parameters );

         SolverMonitorType solverMonitor;
         solverMonitor.setVerbose( verbose );
         if( ! problem.getSolverMonitor() )
            linearSystemSolver.setSolverMonitor( solverMonitor );
         else
            linearSystemSolver.setSolverMonitor( * ( SolverMonitorType* ) problem. getSolverMonitor() );

         TimeStepper timeStepper;
         if( ! timeStepper.setup( parameters ) )
         {
            std::cerr << "The time stepper initiation failed!" << std::endl;
            return false;
         }
         timeStepper.setSolver( linearSystemSolver );

         tnlSolverStarter< ConfigTag > solverStarter;
         return solverStarter.template runPDESolver< Problem, TimeStepper >( problem, parameters, timeStepper );
      };
};






#ifdef UNDEF
template< typename ConfigTag >
   template< typename Problem >
bool tnlSolverStarter< ConfigTag > :: setDiscreteSolver( Problem& problem,
                                                         const Config::ParameterContainer& parameters )
{
   if( ( discreteSolver == "sor" ||
         discreteSolver == "cg" ||
         discreteSolver == "bicg-stab" ||
         discreteSolver == "gmres" ) &&
         timeDiscretisation != "semi-implicit" )
   {
      std::cerr << "The '" << discreteSolver << "' solver can be used only with the semi-implicit time discretisation but not with the "
           <<  timeDiscretisation << " one." << std::endl;
      return false;
   }

   if( discreteSolver == "sor" )
   {
      typedef tnlSORSolver< typename Problem :: DiscreteSolverMatrixType,
                            typename Problem :: DiscreteSolverPreconditioner > DiscreteSolver;
      DiscreteSolver solver;
      double omega = parameters. getParameter< double >( "sor-omega" );
      solver. setOmega( omega );
      //solver. setVerbose( this->verbose );
      return setSemiImplicitTimeDiscretisation< Problem >( problem, parameters, solver );
   }

   if( discreteSolver == "cg" )
   {
      typedef tnlCGSolver< typename Problem :: DiscreteSolverMatrixType,
                           typename Problem :: DiscreteSolverPreconditioner > DiscreteSolver;
      DiscreteSolver solver;
      //solver. setVerbose( this->verbose );
      return setSemiImplicitTimeDiscretisation< Problem >( problem, parameters, solver );
   }

   if( discreteSolver == "bicg-stab" )
   {
      typedef tnlBICGStabSolver< typename Problem :: DiscreteSolverMatrixType,
                                 typename Problem :: DiscreteSolverPreconditioner > DiscreteSolver;
      DiscreteSolver solver;
      //solver. setVerbose( this->verbose );
      return setSemiImplicitTimeDiscretisation< Problem >( problem, parameters, solver );
   }

   if( discreteSolver == "gmres" )
   {
      typedef tnlGMRESSolver< typename Problem :: DiscreteSolverMatrixType,
                              typename Problem :: DiscreteSolverPreconditioner > DiscreteSolver;
      DiscreteSolver solver;
      int restarting = parameters. getParameter< int >( "gmres-restarting" );
      solver. setRestarting( restarting );
      //solver. setVerbose( this->verbose );
      return setSemiImplicitTimeDiscretisation< Problem >( problem, parameters, solver );
   }
   std::cerr << "Unknown discrete solver " << discreteSolver << "." << std::endl;
   return false;
}
#endif

template< typename ConfigTag >
   template< typename Problem,
             typename TimeStepper >
bool tnlSolverStarter< ConfigTag > :: runPDESolver( Problem& problem,
                                                    const Config::ParameterContainer& parameters,
                                                    TimeStepper& timeStepper )
{
   this->totalTimer.reset();
   this->totalTimer.start();
 

   /****
    * Set-up the PDE solver
    */
   tnlPDESolver< Problem, TimeStepper > solver;
   solver.setProblem( problem );
   solver.setTimeStepper( timeStepper );
   if( ! solver.setup( parameters ) )
      return false;

   /****
    * Write a prolog
    */
   int verbose = parameters.getParameter< int >( "verbose" );
   parameters. getParameter< int >( "log-width", logWidth );
   if( verbose )
   {
      Logger logger( logWidth,std::cout );
      solver.writeProlog( logger, parameters );
   }
   String logFileName;
   bool haveLogFile = parameters.getParameter< String >( "log-file", logFileName );
   if( haveLogFile )
   {
      std::fstream logFile;
      logFile.open( logFileName.getString(), std::ios::out );
      if( ! logFile )
      {
         std::cerr << "Unable to open the log file " << logFileName << "." << std::endl;
         return false;
      }
      else
      {
         Logger logger( logWidth, logFile );
         solver.writeProlog( logger, parameters  );
         logFile.close();
      }
   }

   /****
    * Set-up timers
    */
   this->computeTimer.reset();
   this->ioTimer.reset();
   solver.setComputeTimer( this->computeTimer );
   solver.setIoTimer( this->ioTimer );

   /****
    * Start the solver
    */
   bool returnCode( true );
   if( ! solver.solve() )
   {
      returnCode = false;
      if( verbose )
         std::cerr << std::endl << "The solver did not converge. " << std::endl;
      std::fstream logFile;
      logFile.open( logFileName.getString(), std::ios::out | std::ios::app );
      if( ! logFile )
      {
         std::cerr << "Unable to open the log file " << logFileName << "." << std::endl;
         return false;
      }
      else
      {
         logFile << "The solver did not converge. " << std::endl;
         logFile.close();
      }
   }

   /****
    * Stop timers
    */
   this->computeTimer.stop();
   this->totalTimer.stop();

   /****
    * Write an epilog
    */
   if( verbose )
      writeEpilog(std::cout, solver );
   if( haveLogFile )
   {
      std::fstream logFile;
      logFile.open( logFileName.getString(), std::ios::out | std::ios::app );
      if( ! logFile )
      {
         std::cerr << "Unable to open the log file " << logFileName << "." << std::endl;
         return false;
      }
      else
      {
         writeEpilog( logFile, solver );
         logFile.close();
      }
   }
   return returnCode;
}

template< typename ConfigTag >
   template< typename Solver >
bool tnlSolverStarter< ConfigTag > :: writeEpilog( std::ostream& str, const Solver& solver  )
{
   Logger logger( logWidth, str );
   logger.writeSeparator();
   logger.writeCurrentTime( "Finished at:" );
   if( ! solver.writeEpilog( logger ) )
      return false;
   logger.writeParameter< const char* >( "Compute time:", "" );
   this->computeTimer.writeLog( logger, 1 );
   logger.writeParameter< const char* >( "I/O time:", "" );
   this->ioTimer.writeLog( logger, 1 );
   logger.writeParameter< const char* >( "Total time:", "" );
   this->totalTimer.writeLog( logger, 1 );
   char buf[ 256 ];
   sprintf( buf, "%f %%", 100 * ( ( double ) this->totalTimer.getCPUTime() ) / this->totalTimer.getRealTime() );
   logger.writeParameter< char* >( "CPU usage:", buf );
   logger.writeSeparator();
   return true;
}

} // namespace TNL
