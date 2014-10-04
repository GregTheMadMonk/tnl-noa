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
#include <solvers/pde/tnlExplicitTimeStepper.h>
#include <solvers/pde/tnlPDESolver.h>
#include <solvers/tnlIterativeSolverMonitor.h>
#include <solvers/ode/tnlODESolverMonitor.h>

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
          typename ConfigTag,
          bool enabled = tnlConfigTagSemiImplicitSolver< ConfigTag, SemiImplicitSolver >::enabled >
class tnlSolverStarterSemiImplicitSolverSetter{};


template< typename Problem,
          typename ExplicitSolver,
          typename TimeStepper,
          typename ConfigTag >
class tnlSolverStarterExplicitTimeStepperSetter;

template< typename ConfigTag >
tnlSolverStarter< ConfigTag > :: tnlSolverStarter()
: logWidth( 72 )
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
   if( ! problem. init( parameters ) )
   {
      cerr << "The problem initiation failed!" << endl;
      return false;
   }

   /****
    * Set-up the time discretisation
    */
   const tnlString& timeDiscretisation = parameters. GetParameter< tnlString>( "time-discretisation" );
   if( timeDiscretisation == "explicit" )
      return tnlSolverStarterTimeDiscretisationSetter< Problem, tnlExplicitTimeDiscretisationTag, ConfigTag >::run( problem, parameters );
   if( timeDiscretisation == "semi-implicit" )
      return tnlSolverStarterTimeDiscretisationSetter< Problem, tnlSemiImplicitTimeDiscretisationTag, ConfigTag >::run( problem, parameters );
   if( timeDiscretisation == "implicit" )
      return tnlSolverStarterTimeDiscretisationSetter< Problem, tnlImplicitTimeDiscretisationTag, ConfigTag >::run( problem, parameters );
   cerr << "Uknown time discretisation: " << timeDiscretisation << "." << endl;
   return false;
}

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
         cerr << "The time discretisation " << parameters.GetParameter< tnlString >( "time-discretisation" ) << " is not supported." << endl;
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
         const tnlString& discreteSolver = parameters. GetParameter< tnlString>( "discrete-solver" );
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
         const tnlString& discreteSolver = parameters. GetParameter< tnlString>( "discrete-solver" );
         if( discreteSolver == "gmres" )
            return tnlSolverStarterExplicitSolverSetter< Problem, tnlSemiImplicitGMRESSolverTag, ConfigTag >::run( problem, parameters );
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
         const tnlString& discreteSolver = parameters. GetParameter< tnlString>( "discrete-solver" );
         return false;
      }
};

/****
 * Setting the explicit solver
 */

template< typename Problem,
          typename ExplicitSolver,
          typename ConfigTag >
class tnlSolverStarterExplicitSolverSetter< Problem, ExplicitSolver, ConfigTag, false >
{
   public:
      static bool run( Problem& problem,
                       const tnlParameterContainer& parameters )
      {
         cerr << "The explicit solver " << parameters.GetParameter< tnlString >( "discrete-solver" ) << " is not supported." << endl;
         return false;
      }
};

template< typename Problem,
          typename ConfigTag >
class tnlSolverStarterExplicitSolverSetter< Problem, tnlExplicitEulerSolverTag, ConfigTag, true >
{
   public:
      static bool run( Problem& problem,
                       const tnlParameterContainer& parameters )
      {
         typedef tnlExplicitTimeStepper< Problem, tnlEulerSolver > TimeStepper;
         typedef tnlEulerSolver< TimeStepper > ExplicitSolver;
         return tnlSolverStarterExplicitTimeStepperSetter< Problem,
                                                           ExplicitSolver,
                                                           TimeStepper,
                                                           ConfigTag >::run( problem, parameters );
      }
};

template< typename Problem,
          typename ConfigTag >
class tnlSolverStarterExplicitSolverSetter< Problem, tnlExplicitMersonSolverTag, ConfigTag, true >
{
   public:
      static bool run( Problem& problem,
                       const tnlParameterContainer& parameters )
      {
         typedef tnlExplicitTimeStepper< Problem, tnlMersonSolver > TimeStepper;
         typedef tnlMersonSolver< TimeStepper > ExplicitSolver;
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
          typename SemiImplicitSolver,
          typename ConfigTag >
class tnlSolverStarterSemiImplicitSolverSetter< Problem, SemiImplicitSolver, ConfigTag, false >
{
   public:
      static bool run( Problem& problem,
                       const tnlParameterContainer& parameters )
      {
         cerr << "The semi-implicit solver " << parameters.GetParameter< tnlString >( "discrete-solver" ) << " is not supported." << endl;
         return false;
      }
};

template< typename Problem,
          typename ConfigTag >
class tnlSolverStarterSemiImplicitSolverSetter< Problem, tnlSemiImplicitGMRESSolverTag, ConfigTag, true >
{
   public:
      static bool run( Problem& problem,
                       const tnlParameterContainer& parameters )
      {
         typedef tnlSemiImplicitTimeStepper< Problem, tnlGMRESSolver > TimeStepper;
         return tnlSolverStarterSemiImplicitTimeStepperSetter< Problem,
                                                               TimeStepper,
                                                               ConfigTag >::run( problem, parameters );
      }
};

/****
 * Setting the explcicit time stepper
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
         ExplicitSolver explicitSolver;
         explicitSolver.init( parameters );
         int verbose = parameters.GetParameter< int >( "verbose" );
         explicitSolver.setVerbose( verbose );
         tnlODESolverMonitor< typename Problem :: RealType, typename Problem :: IndexType > odeSolverMonitor;
         if( ! problem.getSolverMonitor() )
            explicitSolver.setSolverMonitor( odeSolverMonitor );
         else
            explicitSolver.setSolverMonitor( * ( tnlODESolverMonitor< typename Problem :: RealType, typename Problem :: IndexType >* ) problem. getSolverMonitor() );

         TimeStepper timeStepper;
         if( ! timeStepper.init( parameters ) )
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
      double omega = parameters. GetParameter< double >( "sor-omega" );
      solver. setName( "sor-solver" );
      solver. setOmega( omega );
      //solver. setVerbose( this -> verbose );
      return setSemiImplicitTimeDiscretisation< Problem >( problem, parameters, solver );
   }

   if( discreteSolver == "cg" )
   {
      typedef tnlCGSolver< typename Problem :: DiscreteSolverMatrixType,
                           typename Problem :: DiscreteSolverPreconditioner > DiscreteSolver;
      DiscreteSolver solver;
      solver. setName( "cg-solver" );
      //solver. setVerbose( this -> verbose );
      return setSemiImplicitTimeDiscretisation< Problem >( problem, parameters, solver );
   }

   if( discreteSolver == "bicg-stab" )
   {
      typedef tnlBICGStabSolver< typename Problem :: DiscreteSolverMatrixType,
                                 typename Problem :: DiscreteSolverPreconditioner > DiscreteSolver;
      DiscreteSolver solver;
      solver. setName( "bicg-solver" );
      //solver. setVerbose( this -> verbose );
      return setSemiImplicitTimeDiscretisation< Problem >( problem, parameters, solver );
   }

   if( discreteSolver == "gmres" )
   {
      typedef tnlGMRESSolver< typename Problem :: DiscreteSolverMatrixType,
                              typename Problem :: DiscreteSolverPreconditioner > DiscreteSolver;
      DiscreteSolver solver;
      int restarting = parameters. GetParameter< int >( "gmres-restarting" );
      solver. setName( "gmres-solver" );
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
   this->totalCpuTimer. Reset();
   this->totalRtTimer. Reset();

   /****
    * Set-up the PDE solver
    */
   tnlPDESolver< Problem, TimeStepper > solver;
   solver.setProblem( problem );
   if( ! solver.init( parameters ) )
      return false;
   solver.setTimeStepper( timeStepper );

   /****
    * Write a prolog
    */
   parameters. GetParameter< int >( "log-width", logWidth );
   if( verbose )
   {
      tnlLogger logger( logWidth, cout );
      solver.writeProlog( logger, parameters );
   }
   tnlString logFileName;
   bool haveLogFile = parameters.GetParameter< tnlString >( "log-file", logFileName );
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
   this->computeRtTimer. Reset();
   this->computeCpuTimer. Reset();
   this->ioRtTimer. Reset();
   this->ioRtTimer. Stop();
   this->ioCpuTimer. Reset();
   this->ioCpuTimer. Stop();
   solver.setComputeRtTimer( this -> computeRtTimer );
   solver.setComputeCpuTimer( this -> computeCpuTimer );
   solver.setIoRtTimer( this -> ioRtTimer );
   solver.setIoCpuTimer( this -> ioCpuTimer );

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
   this->computeRtTimer.Stop();
   this->computeCpuTimer.Stop();
   this->totalCpuTimer.Stop();
   this->totalRtTimer.Stop();

   /****
    * Write an epilog
    */
   if( verbose )
      writeEpilog( cout );
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
         writeEpilog( logFile );
         logFile.close();
      }
   }
   return returnCode;
}

template< typename ConfigTag >
bool tnlSolverStarter< ConfigTag > :: writeEpilog( ostream& str )
{
   tnlLogger logger( logWidth, str );
   logger.writeCurrentTime( "Finished at:" );
   logger.writeParameter< double >( "IO Real Time:", this -> ioRtTimer. GetTime() );
   logger.writeParameter< double >( "IO CPU Time:", this -> ioCpuTimer. GetTime() );
   logger.writeParameter< double >( "Compute Real Time:", this -> computeRtTimer. GetTime() );
   logger.writeParameter< double >( "Compute CPU Time:", this -> computeCpuTimer. GetTime() );
   logger.writeParameter< double >( "Total Real Time:", this -> totalRtTimer. GetTime() );
   logger.writeParameter< double >( "Total CPU Time:", this -> totalCpuTimer. GetTime() );
   char buf[ 256 ];
   sprintf( buf, "%f %%", 100 * ( ( double ) this -> totalCpuTimer. GetTime() ) / this -> totalRtTimer. GetTime() );
   logger.writeParameter< char* >( "CPU usage:", buf );
   logger.writeSeparator();
}

#endif /* TNLSOLVERSTARTER_IMPL_H_ */
