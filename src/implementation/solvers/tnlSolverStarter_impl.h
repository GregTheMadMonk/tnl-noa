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

tnlSolverStarter :: tnlSolverStarter()
: logWidth( 72 )
{
}

template< typename Problem >
bool tnlSolverStarter :: run( const tnlParameterContainer& parameters )
{
   this -> verbose = parameters. GetParameter< int >( "verbose" );

   /****
    * Create and set-up the problem
    */
   Problem problem;
   if( ! problem. init( parameters ) )
      return false;

   return setDiscreteSolver< Problem >( problem, parameters );
}

template< typename Problem >
bool tnlSolverStarter :: setDiscreteSolver( Problem& problem,
                                            const tnlParameterContainer& parameters )
{
   const tnlString& discreteSolver = parameters. GetParameter< tnlString>( "discrete-solver" );
   const tnlString& timeDiscretisation = parameters. GetParameter< tnlString>( "time-discretisation" );

   if( timeDiscretisation != "explicit" &&
       timeDiscretisation != "semi-implicit" &&
       timeDiscretisation != "fully-implicit" )
   {
      cerr << "Unknown time discretisation '" << timeDiscretisation << "'." << endl;
      return false;
   }

   if( ( discreteSolver == "euler" || discreteSolver == "merson" ) &&
        timeDiscretisation != "explicit" )
   {
      cerr << "The '" << discreteSolver << "' solver can be used only with the explicit time discretisation but not with the "
           <<  timeDiscretisation << " one." << endl;
      return false;
   }

   if( discreteSolver == "euler" )
   {
      typedef tnlEulerSolver< Problem > DiscreteSolver;
      DiscreteSolver solver;
      solver. setName( "euler-solver" );
      solver. setVerbose( this -> verbose );
      tnlODESolverMonitor< typename Problem :: RealType, typename Problem :: IndexType > odeSolverMonitor;
      if( ! problem. getSolverMonitor() )
         solver. setSolverMonitor( odeSolverMonitor );
      else
         solver. setSolverMonitor( * ( tnlODESolverMonitor< typename Problem :: RealType, typename Problem :: IndexType >* ) problem. getSolverMonitor() );
      return setExplicitTimeDiscretisation( problem, parameters, solver );
   }

   if( discreteSolver == "merson" )
   {
      typedef tnlMersonSolver< Problem > DiscreteSolver;
      DiscreteSolver solver;
      double adaptivity = parameters. GetParameter< double >( "merson-adaptivity" );
      solver. setName( "merson-solver" );
      solver. setAdaptivity( adaptivity );
      solver. setVerbose( this -> verbose );
      tnlODESolverMonitor< typename Problem :: RealType, typename Problem :: IndexType > odeSolverMonitor;
      if( ! problem. getSolverMonitor() )
         solver. setSolverMonitor( odeSolverMonitor );
      else
         solver. setSolverMonitor( * ( tnlODESolverMonitor< typename Problem :: RealType, typename Problem :: IndexType >* ) problem. getSolverMonitor() );

      return setExplicitTimeDiscretisation( problem, parameters, solver );
   }

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

template< typename Problem,
          template < typename > class DiscreteSolver >
bool tnlSolverStarter :: setExplicitTimeDiscretisation( Problem& problem,
                                                        const tnlParameterContainer& parameters,
                                                        DiscreteSolver< Problem >& discreteSolver )
{
   typedef tnlExplicitTimeStepper< Problem, DiscreteSolver > TimeStepperType;
   TimeStepperType timeStepper;
   timeStepper. setSolver( discreteSolver );
   timeStepper. setTau( parameters. GetParameter< double >( "initial-tau" ) );
   return runPDESolver< Problem, TimeStepperType >( problem, parameters, timeStepper );
}

template< typename Problem,
          typename DiscreteSolver >
bool tnlSolverStarter :: setSemiImplicitTimeDiscretisation( Problem& problem,
                                                            const tnlParameterContainer& parameters,
                                                            DiscreteSolver& discreteSolver )
{

}

template< typename Problem >
bool tnlSolverStarter :: writeProlog( ostream& str,
                                      const tnlParameterContainer& parameters,
                                      const Problem& problem )
{
   parameters. GetParameter< int >( "log-width", logWidth );
   tnlLogger logger( logWidth, str );
   logger. WriteHeader( problem. getPrologHeader() );
   problem. writeProlog( logger, parameters );
   logger. WriteSeparator();
   logger. WriteParameter< tnlString >( "Time discretisation:", "time-discretisation", parameters );
   logger. WriteParameter< double >( "Initial tau:", "initial-tau", parameters );
   logger. WriteParameter< double >( "Final time:", "final-time", parameters );
   logger. WriteParameter< double >( "Snapshot period:", "snapshot-period", parameters );
   const tnlString& solverName = parameters. GetParameter< tnlString >( "discrete-solver" );
   logger. WriteParameter< tnlString >( "Discrete solver:", "discrete-solver", parameters );
   if( solverName == "merson" )
      logger. WriteParameter< double >( "Adaptivity:", "merson-adaptivity", parameters, 1 );
   if( solverName == "sor" )
      logger. WriteParameter< double >( "Omega:", "sor-omega", parameters, 1 );
   if( solverName == "gmres" )
      logger. WriteParameter< int >( "Restarting:", "gmres-restarting", parameters, 1 );
   logger. WriteSeparator();
   logger. WriteParameter< tnlString >( "Real type:", "real-type", parameters, 0 );
   logger. WriteParameter< tnlString >( "Index type:", "index-type", parameters, 0 );
   logger. WriteParameter< tnlString >( "Device:", "device", parameters, 0 );
   logger. WriteSeparator();
   logger. writeSystemInformation();
   logger. WriteSeparator();
   logger. writeCurrentTime( "Started at:" );
   return true;
}

template< typename Problem,
          typename TimeStepper >
bool tnlSolverStarter :: runPDESolver( Problem& problem,
                                       const tnlParameterContainer& parameters,
                                       TimeStepper& timeStepper )
{
   this -> totalCpuTimer. Reset();
   this -> totalRtTimer. Reset();

   /***
    * Set-up the initial condition
    */
   typedef typename Problem :: DofVectorType DofVectorType;
   if( ! problem. setInitialCondition( parameters ) )
      return false;

   /****
    * Set-up the PDE solver
    */
   tnlPDESolver< Problem, TimeStepper > solver;
   solver. setProblem( problem );
   solver. setTimeStepper( timeStepper );
   solver. setSnapshotTau( parameters. GetParameter< double >( "snapshot-period" ) );
   solver. setFinalTime( parameters. GetParameter< double >( "final-time" ) );

   /****
    * Write a prolog
    */
   if( verbose )
      writeProlog( cout, parameters, problem );
   tnlString logFileName;
   if( parameters. GetParameter< tnlString >( "log-file", logFileName ) )
   {
      fstream logFile;
      logFile. open( logFileName. getString(), ios :: out );
      if( ! logFile )
      {
         cerr << "Unable to open the log file " << logFileName << "." << endl;
         return false;
      }
      else
      {
         writeProlog( logFile, parameters, problem  );
         logFile. close();
      }
   }

   /****
    * Set-up timers
    */
   this -> computeRtTimer. Reset();
   this -> computeCpuTimer. Reset();
   this -> ioRtTimer. Reset();
   this -> ioRtTimer. Stop();
   this -> ioCpuTimer. Reset();
   this -> ioCpuTimer. Stop();
   solver. setComputeRtTimer( this -> computeRtTimer );
   solver. setComputeCpuTimer( this -> computeCpuTimer );
   solver. setIoRtTimer( this -> ioRtTimer );
   solver. setIoCpuTimer( this -> ioCpuTimer );

   /****
    * Start the solver
    */
   if( ! solver. solve() )
      return false;

   /****
    * Stop timers
    */
   this -> computeRtTimer. Stop();
   this -> computeCpuTimer. Stop();
   this -> totalCpuTimer. Stop();
   this -> totalRtTimer. Stop();

   /****
    * Write an epilog
    */
   if( verbose )
      writeEpilog( cout );
   if( parameters. GetParameter< tnlString >( "log-file", logFileName ) )
   {
      fstream logFile;
      logFile. open( logFileName. getString(), ios :: out | ios :: app );
      if( ! logFile )
      {
         cerr << "Unable to open the log file " << logFileName << "." << endl;
         return false;
      }
      else
      {
         writeEpilog( logFile );
         logFile. close();
      }
   }
   return true;
}

bool tnlSolverStarter :: writeEpilog( ostream& str )
{
   tnlLogger logger( logWidth, str );
   logger. writeCurrentTime( "Finished at:" );
   logger. WriteParameter< double >( "IO Real Time:", this -> ioRtTimer. GetTime() );
   logger. WriteParameter< double >( "IO CPU Time:", this -> ioCpuTimer. GetTime() );
   logger. WriteParameter< double >( "Compute Real Time:", this -> computeRtTimer. GetTime() );
   logger. WriteParameter< double >( "Compute CPU Time:", this -> computeCpuTimer. GetTime() );
   logger. WriteParameter< double >( "Total Real Time:", this -> totalRtTimer. GetTime() );
   logger. WriteParameter< double >( "Total CPU Time:", this -> totalCpuTimer. GetTime() );
   char buf[ 256 ];
   sprintf( buf, "%f %%", 100 * ( ( double ) this -> totalCpuTimer. GetTime() ) / this -> totalRtTimer. GetTime() );
   logger. WriteParameter< char* >( "CPU usage:", buf );
   logger. WriteSeparator();
}

#endif /* TNLSOLVERSTARTER_IMPL_H_ */
