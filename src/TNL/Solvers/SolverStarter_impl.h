/***************************************************************************
                          SolverStarter_impl.h  -  description
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
#include <TNL/Devices/Cuda.h>
#include <TNL/Solvers/ODE/Merson.h>
#include <TNL/Solvers/ODE/Euler.h>
#include <TNL/Solvers/Linear/SOR.h>
#include <TNL/Solvers/Linear/CG.h>
#include <TNL/Solvers/Linear/BICGStab.h>
#include <TNL/Solvers/Linear/GMRES.h>
#include <TNL/Solvers/Linear/TFQMR.h>
#include <TNL/Solvers/Linear/UmfpackWrapper.h>
#include <TNL/Solvers/Linear/Preconditioners/Dummy.h>
#include <TNL/Solvers/Linear/Preconditioners/Diagonal.h>
#include <TNL/Solvers/PDE/ExplicitTimeStepper.h>
#include <TNL/Solvers/PDE/SemiImplicitTimeStepper.h>
#include <TNL/Solvers/PDE/PDESolver.h>
#include <TNL/Solvers/ODE/ODESolverMonitor.h>

namespace TNL {
namespace Solvers {   

template< typename Problem,
          typename ConfigTag,
          typename TimeStepper = typename Problem::TimeStepper >
class tnlUserDefinedTimeDiscretisationSetter;

template< typename Problem,
          typename TimeDiscretisation,
          typename ConfigTag,
          bool enabled = ConfigTagTimeDiscretisation< ConfigTag, TimeDiscretisation >::enabled >
class SolverStarterTimeDiscretisationSetter{};

template< typename Problem,
          typename ExplicitSolver,
          typename ConfigTag,
          bool enabled = ConfigTagExplicitSolver< ConfigTag, ExplicitSolver >::enabled >
class SolverStarterExplicitSolverSetter{};

template< typename Problem,
          typename SemiImplicitSolver,
          template<typename, typename, typename> class Preconditioner,
          typename ConfigTag,
          bool enabled = ConfigTagSemiImplicitSolver< ConfigTag, SemiImplicitSolver >::enabled >
class SolverStarterSemiImplicitSolverSetter{};


template< typename Problem,
          typename SemiImplicitSolverTag,
          typename ConfigTag >
class SolverStarterPreconditionerSetter;

template< typename Problem,
          typename ExplicitSolver,
          typename TimeStepper,
          typename ConfigTag >
class SolverStarterExplicitTimeStepperSetter;

template< typename Problem,
          typename TimeStepper,
          typename ConfigTag >
class SolverStarterSemiImplicitTimeStepperSetter;

template< typename ConfigTag >
SolverStarter< ConfigTag > :: SolverStarter()
: logWidth( 80 )
{
}

template< typename ConfigTag >
   template< typename Problem >
bool SolverStarter< ConfigTag > :: run( const Config::ParameterContainer& parameters )
{
   /****
    * Create and set-up the problem
    */
   if( ! Devices::Host::setup( parameters ) ||
       ! Devices::Cuda::setup( parameters ) )
      return false;
   Problem problem;
   /*if( ! problem.setup( parameters ) )
   {
      std::cerr << "The problem initiation failed!" << std::endl;
      return false;
   }*/

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
         SolverStarter< ConfigTag > solverStarter;
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
            return SolverStarterTimeDiscretisationSetter< Problem, ExplicitTimeDiscretisationTag, ConfigTag >::run( problem, parameters );
         if( timeDiscretisation == "semi-implicit" )
            return SolverStarterTimeDiscretisationSetter< Problem, SemiImplicitTimeDiscretisationTag, ConfigTag >::run( problem, parameters );
         if( timeDiscretisation == "implicit" )
            return SolverStarterTimeDiscretisationSetter< Problem, ImplicitTimeDiscretisationTag, ConfigTag >::run( problem, parameters );
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
class SolverStarterTimeDiscretisationSetter< Problem, TimeDiscretisation, ConfigTag, false >
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
class SolverStarterTimeDiscretisationSetter< Problem, ExplicitTimeDiscretisationTag, ConfigTag, true >
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
            return SolverStarterExplicitSolverSetter< Problem, ExplicitEulerSolverTag, ConfigTag >::run( problem, parameters );
         if( discreteSolver == "merson" )
            return SolverStarterExplicitSolverSetter< Problem, ExplicitMersonSolverTag, ConfigTag >::run( problem, parameters );
         return false;
      }
};

template< typename Problem,
          typename ConfigTag >
class SolverStarterTimeDiscretisationSetter< Problem, SemiImplicitTimeDiscretisationTag, ConfigTag, true >
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
            return SolverStarterPreconditionerSetter< Problem, SemiImplicitSORSolverTag, ConfigTag >::run( problem, parameters );
         if( discreteSolver == "cg" )
            return SolverStarterPreconditionerSetter< Problem, SemiImplicitCGSolverTag, ConfigTag >::run( problem, parameters );
         if( discreteSolver == "bicgstab" )
            return SolverStarterPreconditionerSetter< Problem, SemiImplicitBICGStabSolverTag, ConfigTag >::run( problem, parameters );
         if( discreteSolver == "gmres" )
            return SolverStarterPreconditionerSetter< Problem, SemiImplicitGMRESSolverTag, ConfigTag >::run( problem, parameters );
         if( discreteSolver == "tfqmr" )
            return SolverStarterPreconditionerSetter< Problem, SemiImplicitTFQMRSolverTag, ConfigTag >::run( problem, parameters );
#ifdef HAVE_UMFPACK
         if( discreteSolver == "umfpack" )
            return SolverStarterPreconditionerSetter< Problem, SemiImplicitUmfpackSolverTag, ConfigTag >::run( problem, parameters );
#endif
         return false;
      }
};

template< typename Problem,
          typename ConfigTag >
class SolverStarterTimeDiscretisationSetter< Problem, ImplicitTimeDiscretisationTag, ConfigTag, true >
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
class SolverStarterExplicitSolverSetter< Problem, ExplicitSolverTag, ConfigTag, false >
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
class SolverStarterExplicitSolverSetter< Problem, ExplicitSolverTag, ConfigTag, true >
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         typedef PDE::ExplicitTimeStepper< Problem, ExplicitSolverTag::template Template > TimeStepper;
         typedef typename ExplicitSolverTag::template Template< TimeStepper > ExplicitSolver;
         return SolverStarterExplicitTimeStepperSetter< Problem,
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
class SolverStarterPreconditionerSetter
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         const String& preconditioner = parameters.getParameter< String>( "preconditioner" );

         if( preconditioner == "none" )
            return SolverStarterSemiImplicitSolverSetter< Problem, SemiImplicitSolverTag, Linear::Preconditioners::Dummy, ConfigTag >::run( problem, parameters );
         if( preconditioner == "diagonal" )
            return SolverStarterSemiImplicitSolverSetter< Problem, SemiImplicitSolverTag, Linear::Preconditioners::Diagonal, ConfigTag >::run( problem, parameters );

         std::cerr << "Unknown preconditioner " << preconditioner << ". It can be only: none, diagonal." << std::endl;
         return false;
      }
};

template< typename Problem,
          typename SemiImplicitSolverTag,
          template<typename, typename, typename> class Preconditioner,
          typename ConfigTag >
class SolverStarterSemiImplicitSolverSetter< Problem, SemiImplicitSolverTag, Preconditioner, ConfigTag, false >
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
class SolverStarterSemiImplicitSolverSetter< Problem, SemiImplicitSolverTag, Preconditioner, ConfigTag, true >
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
         typedef PDE::SemiImplicitTimeStepper< Problem, LinearSystemSolver > TimeStepper;
         return SolverStarterSemiImplicitTimeStepperSetter< Problem,
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
class SolverStarterExplicitTimeStepperSetter
{
   public:

      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters)
      {
         typedef typename Problem::RealType RealType;
         typedef typename Problem::IndexType IndexType;
         typedef ODE::ODESolverMonitor< RealType, IndexType > SolverMonitorType;

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

         SolverStarter< ConfigTag > solverStarter;
         return solverStarter.template runPDESolver< Problem, TimeStepper >( problem, parameters, timeStepper );
      };
};

/****
 * Setting the semi-implicit time stepper
 */
template< typename Problem,
          typename TimeStepper,
          typename ConfigTag >
class SolverStarterSemiImplicitTimeStepperSetter
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
         typedef IterativeSolverMonitor< RealType, IndexType > SolverMonitorType;

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

         SolverStarter< ConfigTag > solverStarter;
         return solverStarter.template runPDESolver< Problem, TimeStepper >( problem, parameters, timeStepper );
      };
};






#ifdef UNDEF
template< typename ConfigTag >
   template< typename Problem >
bool SolverStarter< ConfigTag > :: setDiscreteSolver( Problem& problem,
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
      typedef SOR< typename Problem :: DiscreteSolverMatrixType,
                            typename Problem :: DiscreteSolverPreconditioner > DiscreteSolver;
      DiscreteSolver solver;
      double omega = parameters. getParameter< double >( "sor-omega" );
      solver. setOmega( omega );
      //solver. setVerbose( this->verbose );
      return setSemiImplicitTimeDiscretisation< Problem >( problem, parameters, solver );
   }

   if( discreteSolver == "cg" )
   {
      typedef CG< typename Problem :: DiscreteSolverMatrixType,
                           typename Problem :: DiscreteSolverPreconditioner > DiscreteSolver;
      DiscreteSolver solver;
      //solver. setVerbose( this->verbose );
      return setSemiImplicitTimeDiscretisation< Problem >( problem, parameters, solver );
   }

   if( discreteSolver == "bicg-stab" )
   {
      typedef BICGStab< typename Problem :: DiscreteSolverMatrixType,
                                 typename Problem :: DiscreteSolverPreconditioner > DiscreteSolver;
      DiscreteSolver solver;
      //solver. setVerbose( this->verbose );
      return setSemiImplicitTimeDiscretisation< Problem >( problem, parameters, solver );
   }

   if( discreteSolver == "gmres" )
   {
      typedef GMRES< typename Problem :: DiscreteSolverMatrixType,
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
bool SolverStarter< ConfigTag > :: runPDESolver( Problem& problem,
                                                    const Config::ParameterContainer& parameters,
                                                    TimeStepper& timeStepper )
{
   this->totalTimer.reset();
   this->totalTimer.start();
 

   /****
    * Set-up the PDE solver
    */
   PDE::PDESolver< Problem, TimeStepper > solver;
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
bool SolverStarter< ConfigTag > :: writeEpilog( std::ostream& str, const Solver& solver  )
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

} // namespace Solvers
} // namespace TNL
