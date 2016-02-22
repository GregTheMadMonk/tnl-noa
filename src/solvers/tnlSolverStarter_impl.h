/***************************************************************************
                          tnlSolverStarter_impl.h  -  description
                             -------------------
    begin                : Mar 9, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLSOLVERSTARTER_IMPL_H_
#define TNLSOLVERSTARTER_IMPL_H_

#include <tnlConfig.h>
#include <core/tnlLogger.h>
#include <core/tnlString.h>
#include <solvers/ode/tnlMersonSolver.h>
#include <solvers/ode/tnlEulerSolver.h>
#include <solvers/linear/stationary/tnlSORSolver.h>
#include <solvers/linear/krylov/tnlCGSolver.h>
#include <solvers/linear/krylov/tnlBICGStabSolver.h>
#include <solvers/linear/krylov/tnlGMRESSolver.h>
#include <solvers/linear/krylov/tnlTFQMRSolver.h>
#include <solvers/preconditioners/tnlDummyPreconditioner.h>
#include <solvers/preconditioners/tnlDiagonalPreconditioner.h>
#include <solvers/pde/tnlExplicitTimeStepper.h>
#include <solvers/pde/tnlSemiImplicitTimeStepper.h>
#include <solvers/pde/tnlPDESolver.h>
#include <solvers/tnlIterativeSolverMonitor.h>
#include <solvers/ode/tnlODESolverMonitor.h>

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
bool tnlSolverStarter< ConfigTag > :: run( const tnlParameterContainer& parameters )
{
   /****
    * Create and set-up the problem
    */
   Problem problem;
   if( ! problem.setup( parameters ) )
   {
      cerr << "The problem initiation failed!" << endl;
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
                       const tnlParameterContainer& parameters )
      {
         TimeStepper timeStepper;
         if( ! timeStepper.setup( parameters ) )
         {
            cerr << "The time stepper initiation failed!" << endl;
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
                       const tnlParameterContainer& parameters )
      {
         /****
          * Set-up the time discretisation
          */
         const tnlString& timeDiscretisation = parameters. getParameter< tnlString>( "time-discretisation" );
         if( timeDiscretisation == "explicit" )
            return tnlSolverStarterTimeDiscretisationSetter< Problem, tnlExplicitTimeDiscretisationTag, ConfigTag >::run( problem, parameters );
         if( timeDiscretisation == "semi-implicit" )
            return tnlSolverStarterTimeDiscretisationSetter< Problem, tnlSemiImplicitTimeDiscretisationTag, ConfigTag >::run( problem, parameters );
         if( timeDiscretisation == "implicit" )
            return tnlSolverStarterTimeDiscretisationSetter< Problem, tnlImplicitTimeDiscretisationTag, ConfigTag >::run( problem, parameters );
         cerr << "Uknown time discretisation: " << timeDiscretisation << "." << endl;
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
                       const tnlParameterContainer& parameters )
      {
         cerr << "The time discretisation " << parameters.getParameter< tnlString >( "time-discretisation" ) << " is not supported." << endl;
         return false;
      }
};

template< typename Problem,
          typename ConfigTag >
class tnlSolverStarterTimeDiscretisationSetter< Problem, tnlExplicitTimeDiscretisationTag, ConfigTag, true >
{
   public:
      static bool run( Problem& problem,
                       const tnlParameterContainer& parameters )
      {
         const tnlString& discreteSolver = parameters. getParameter< tnlString>( "discrete-solver" );
         if( discreteSolver != "euler" &&
             discreteSolver != "merson" )
         {
            cerr << "Unknown explicit discrete solver " << discreteSolver << ". It can be only: euler or merson." << endl;
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
                       const tnlParameterContainer& parameters )
      {
         const tnlString& discreteSolver = parameters. getParameter< tnlString>( "discrete-solver" );
         if( discreteSolver != "sor" &&
             discreteSolver != "cg" &&
             discreteSolver != "bicgstab" &&
             discreteSolver != "gmres" &&
             discreteSolver != "tfqmr" )
         {
            cerr << "Unknown semi-implicit discrete solver " << discreteSolver << ". It can be only: sor, cg, bicgstab, gmres or tfqmr." << endl;
            return false;
         }

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
         return false;
      }
};

template< typename Problem,
          typename ConfigTag >
class tnlSolverStarterTimeDiscretisationSetter< Problem, tnlImplicitTimeDiscretisationTag, ConfigTag, true >
{
   public:
      static bool run( Problem& problem,
                       const tnlParameterContainer& parameters )
      {
         const tnlString& discreteSolver = parameters. getParameter< tnlString>( "discrete-solver" );
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
                       const tnlParameterContainer& parameters )
      {
         cerr << "The explicit solver " << parameters.getParameter< tnlString >( "discrete-solver" ) << " is not supported." << endl;
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
                       const tnlParameterContainer& parameters )
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
                       const tnlParameterContainer& parameters )
      {
         const tnlString& preconditioner = parameters.getParameter< tnlString>( "preconditioner" );

         if( preconditioner == "none" )
            return tnlSolverStarterSemiImplicitSolverSetter< Problem, SemiImplicitSolverTag, tnlDummyPreconditioner, ConfigTag >::run( problem, parameters );
         if( preconditioner == "diagonal" )
            return tnlSolverStarterSemiImplicitSolverSetter< Problem, SemiImplicitSolverTag, tnlDiagonalPreconditioner, ConfigTag >::run( problem, parameters );

         cerr << "Unknown preconditioner " << preconditioner << ". It can be only: none, diagonal." << endl;
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
                       const tnlParameterContainer& parameters )
      {
         cerr << "The semi-implicit solver " << parameters.getParameter< tnlString >( "discrete-solver" ) << " is not supported." << endl;
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
                       const tnlParameterContainer& parameters )
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
                       const tnlParameterContainer& parameters)
      {
         typedef typename Problem::RealType RealType;
         typedef typename Problem::IndexType IndexType;
         typedef tnlODESolverMonitor< RealType, IndexType > SolverMonitorType;

         ExplicitSolver explicitSolver;
         explicitSolver.setup( parameters );
         int verbose = parameters.getParameter< int >( "verbose" );
         explicitSolver.setVerbose( verbose );
         SolverMonitorType odeSolverMonitor;
         if( ! problem.getSolverMonitor() )
            explicitSolver.setSolverMonitor( odeSolverMonitor );
         else
            explicitSolver.setSolverMonitor( * ( SolverMonitorType* ) problem. getSolverMonitor() );

         TimeStepper timeStepper;
         if( ! timeStepper.setup( parameters ) )
         {
            cerr << "The time stepper initiation failed!" << endl;
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
                       const tnlParameterContainer& parameters)
      {
         typedef typename TimeStepper::LinearSystemSolverType LinearSystemSolverType;
         typedef typename LinearSystemSolverType::PreconditionerType PreconditionerType;
         typedef typename LinearSystemSolverType::MatrixType MatrixType;
         typedef typename Problem::RealType RealType;
         typedef typename Problem::IndexType IndexType;
         typedef tnlIterativeSolverMonitor< RealType, IndexType > SolverMonitorType;

         LinearSystemSolverType linearSystemSolver;
         linearSystemSolver.setup( parameters );

         SolverMonitorType solverMonitor;
         if( ! problem.getSolverMonitor() )
            linearSystemSolver.setSolverMonitor( solverMonitor );
         else
            linearSystemSolver.setSolverMonitor( * ( SolverMonitorType* ) problem. getSolverMonitor() );

         TimeStepper timeStepper;
         if( ! timeStepper.setup( parameters ) )
         {
            cerr << "The time stepper initiation failed!" << endl;
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
                                                         const tnlParameterContainer& parameters )
{
   if( ( discreteSolver == "sor" ||
         discreteSolver == "cg" ||
         discreteSolver == "bicg-stab" ||
         discreteSolver == "gmres" ) &&
         timeDiscretisation != "semi-implicit" )
   {
      cerr << "The '" << discreteSolver << "' solver can be used only with the semi-implicit time discretisation but not with the "
           <<  timeDiscretisation << " one." << endl;
      return false;
   }

   if( discreteSolver == "sor" )
   {
      typedef tnlSORSolver< typename Problem :: DiscreteSolverMatrixType,
                            typename Problem :: DiscreteSolverPreconditioner > DiscreteSolver;
      DiscreteSolver solver;
      double omega = parameters. getParameter< double >( "sor-omega" );
      solver. setOmega( omega );
      //solver. setVerbose( this -> verbose );
      return setSemiImplicitTimeDiscretisation< Problem >( problem, parameters, solver );
   }

   if( discreteSolver == "cg" )
   {
      typedef tnlCGSolver< typename Problem :: DiscreteSolverMatrixType,
                           typename Problem :: DiscreteSolverPreconditioner > DiscreteSolver;
      DiscreteSolver solver;
      //solver. setVerbose( this -> verbose );
      return setSemiImplicitTimeDiscretisation< Problem >( problem, parameters, solver );
   }

   if( discreteSolver == "bicg-stab" )
   {
      typedef tnlBICGStabSolver< typename Problem :: DiscreteSolverMatrixType,
                                 typename Problem :: DiscreteSolverPreconditioner > DiscreteSolver;
      DiscreteSolver solver;
      //solver. setVerbose( this -> verbose );
      return setSemiImplicitTimeDiscretisation< Problem >( problem, parameters, solver );
   }

   if( discreteSolver == "gmres" )
   {
      typedef tnlGMRESSolver< typename Problem :: DiscreteSolverMatrixType,
                              typename Problem :: DiscreteSolverPreconditioner > DiscreteSolver;
      DiscreteSolver solver;
      int restarting = parameters. getParameter< int >( "gmres-restarting" );
      solver. setRestarting( restarting );
      //solver. setVerbose( this -> verbose );
      return setSemiImplicitTimeDiscretisation< Problem >( problem, parameters, solver );
   }
   cerr << "Unknown discrete solver " << discreteSolver << "." << endl;
   return false;
}
#endif

template< typename ConfigTag >
   template< typename Problem,
             typename TimeStepper >
bool tnlSolverStarter< ConfigTag > :: runPDESolver( Problem& problem,
                                                    const tnlParameterContainer& parameters,
                                                    TimeStepper& timeStepper )
{
   this->totalCpuTimer.reset();
   this->totalRtTimer.reset();

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
      tnlLogger logger( logWidth, cout );
      solver.writeProlog( logger, parameters );
   }
   tnlString logFileName;
   bool haveLogFile = parameters.getParameter< tnlString >( "log-file", logFileName );
   if( haveLogFile )
   {
      fstream logFile;
      logFile.open( logFileName.getString(), ios :: out );
      if( ! logFile )
      {
         cerr << "Unable to open the log file " << logFileName << "." << endl;
         return false;
      }
      else
      {
         tnlLogger logger( logWidth, logFile );
         solver.writeProlog( logger, parameters  );
         logFile.close();
      }
   }

   /****
    * Set-up timers
    */
   this->computeRtTimer.reset();
   this->computeCpuTimer.reset();
   this->ioRtTimer.reset();
   this->ioRtTimer.stop();
   this->ioCpuTimer.reset();
   this->ioCpuTimer.stop();
   solver.setComputeRtTimer( this->computeRtTimer );
   solver.setComputeCpuTimer( this->computeCpuTimer );
   solver.setIoRtTimer( this->ioRtTimer );
   solver.setIoCpuTimer( this->ioCpuTimer );

   /****
    * Start the solver
    */
   bool returnCode( true );
   if( ! solver.solve() )
   {
      returnCode = false;
      if( verbose )
         cerr << endl << "The solver did not converge. " << endl;
      fstream logFile;
      logFile.open( logFileName.getString(), ios::out | ios::app );
      if( ! logFile )
      {
         cerr << "Unable to open the log file " << logFileName << "." << endl;
         return false;
      }
      else
      {
         logFile << "The solver did not converge. " << endl;
         logFile.close();
      }
   }

   /****
    * Stop timers
    */
   this->computeRtTimer.stop();
   this->computeCpuTimer.stop();
   this->totalCpuTimer.stop();
   this->totalRtTimer.stop();

   /****
    * Write an epilog
    */
   if( verbose )
      writeEpilog( cout, solver );
   if( haveLogFile )
   {
      fstream logFile;
      logFile.open( logFileName.getString(), ios::out | ios::app );
      if( ! logFile )
      {
         cerr << "Unable to open the log file " << logFileName << "." << endl;
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
bool tnlSolverStarter< ConfigTag > :: writeEpilog( ostream& str, const Solver& solver  )
{
   tnlLogger logger( logWidth, str );
   logger.writeCurrentTime( "Finished at:" );
   if( ! solver.writeEpilog( logger ) )
      return false;
   logger.writeParameter< double >( "IO Real Time:", this -> ioRtTimer. getTime() );
   logger.writeParameter< double >( "IO CPU Time:", this -> ioCpuTimer. getTime() );
   logger.writeParameter< double >( "Compute Real Time:", this -> computeRtTimer. getTime() );
   logger.writeParameter< double >( "Compute CPU Time:", this -> computeCpuTimer. getTime() );
   logger.writeParameter< double >( "Total Real Time:", this -> totalRtTimer. getTime() );
   logger.writeParameter< double >( "Total CPU Time:", this -> totalCpuTimer. getTime() );
   char buf[ 256 ];
   sprintf( buf, "%f %%", 100 * ( ( double ) this -> totalCpuTimer. getTime() ) / this -> totalRtTimer. getTime() );
   logger.writeParameter< char* >( "CPU usage:", buf );
   logger.writeSeparator();
   return true;
}

#endif /* TNLSOLVERSTARTER_IMPL_H_ */
