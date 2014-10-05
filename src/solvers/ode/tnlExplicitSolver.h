/***************************************************************************
                          tnlExplicitSolver.h  -  description
                             -------------------
    begin                : 2007/06/17
    copyright            : (C) 2007 by Tomas Oberhuber
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

#ifndef tnlExplicitSolverH
#define tnlExplicitSolverH

#include <iomanip>
#include <core/tnlTimerCPU.h>
#include <core/tnlTimerRT.h>
#include <core/tnlFlopsCounter.h>
#include <core/tnlObject.h>
#include <solvers/ode/tnlODESolverMonitor.h>
#include <solvers/tnlIterativeSolver.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>

template< class Problem >
class tnlExplicitSolver : public tnlIterativeSolver< typename Problem::RealType,
                                                     typename Problem::IndexType >
{
   public:
   
   typedef Problem ProblemType;
   typedef typename Problem :: DofVectorType DofVectorType;
   typedef typename Problem :: RealType RealType;
   typedef typename Problem :: DeviceType DeviceType;
   typedef typename Problem :: IndexType IndexType;

   tnlExplicitSolver();

   static void configSetup( tnlConfigDescription& config,
                            const tnlString& prefix = "" );

   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );

   void setProblem( Problem& problem );

   void setTime( const RealType& t );

   const RealType& getTime() const;
   
   void setStopTime( const RealType& stopTime );

   RealType getStopTime() const;

   void setTau( const RealType& t );
   
   const RealType& getTau() const;

   void setMPIComm( MPI_Comm comm );
  
   void setVerbose( IndexType v );

   void setTimerCPU( tnlTimerCPU* timer );

   void setTimerRT( tnlTimerRT* timer );
   
   virtual bool solve( DofVectorType& u ) = 0;

   void setTestingMode( bool testingMode );

   void setRefreshRate( const IndexType& refreshRate );

   void setSolverMonitor( tnlODESolverMonitor< RealType, IndexType >& solverMonitor );

   void refreshSolverMonitor();

protected:
    
   /****
    * Current time of the parabolic problem.
    */
   RealType time;

   /****
    * The method solve will stop when reaching the stopTime.
    */
   RealType stopTime;

   /****
    * Current time step.
    */
   RealType tau;

   MPI_Comm solver_comm;

   IndexType verbosity;

   tnlTimerCPU* cpu_timer;
   
   tnlTimerRT* rt_timer;

   bool testingMode;

   Problem* problem;

   tnlODESolverMonitor< RealType, IndexType >* solverMonitor;
};

template< typename Problem >
tnlExplicitSolver < Problem > :: tnlExplicitSolver()
:  time( 0.0 ),
   tau( 0.0 ),
   stopTime( 0.0 ),
   solver_comm( MPI_COMM_WORLD ),
   verbosity( 0 ),
   cpu_timer( &default_mcore_cpu_timer ),
   rt_timer( &defaultRTTimer ),
   testingMode( false ),
   problem( 0 ),
   solverMonitor( 0 )
   {
   };

template< typename Problem >
void tnlExplicitSolver < Problem > :: configSetup( tnlConfigDescription& config,
                                                   const tnlString& prefix )
{
   tnlIterativeSolver< typename Problem::RealType, typename Problem::IndexType >::configSetup( config, prefix );
}

template< typename Problem >
bool tnlExplicitSolver < Problem >::setup( const tnlParameterContainer& parameters,
                                          const tnlString& prefix )
{   
   return tnlIterativeSolver< typename Problem::RealType, typename Problem::IndexType >::setup( parameters, prefix );
}


template< typename Problem >
void tnlExplicitSolver< Problem > :: setProblem( Problem& problem )
{
   this -> problem = &problem;
};

template< class Problem >
void tnlExplicitSolver < Problem > :: setTime( const RealType& t )
{
   time = t;
};

template< class Problem >
const typename Problem :: RealType& tnlExplicitSolver < Problem > :: getTime() const
{
   return time;
};

template< class Problem >
void tnlExplicitSolver < Problem > :: setTau( const RealType& t )
{
   tau = t;
};

template< class Problem >
const typename Problem :: RealType& tnlExplicitSolver < Problem > :: getTau() const
{
   return tau;
};

template< class Problem >
typename Problem :: RealType tnlExplicitSolver < Problem > :: getStopTime() const
{
    return stopTime;
}

template< class Problem >
void tnlExplicitSolver < Problem > :: setStopTime( const RealType& stopTime )
{
    this -> stopTime = stopTime;
}

template< class Problem >
void tnlExplicitSolver < Problem > :: setMPIComm( MPI_Comm comm )
{
   solver_comm = comm;
};

template< class Problem >
void tnlExplicitSolver < Problem > :: setVerbose( IndexType v )
{
   verbosity = v;
};

template< class Problem >
void tnlExplicitSolver < Problem > :: setTimerCPU( tnlTimerCPU* timer )
{
   cpu_timer = timer;
};

template< class Problem >
void tnlExplicitSolver < Problem > :: setTimerRT( tnlTimerRT* timer )
{
   rt_timer = timer;
};

template< class Problem >
void tnlExplicitSolver < Problem > :: setSolverMonitor( tnlODESolverMonitor< RealType, IndexType >& solverMonitor )
{
   this -> solverMonitor = &solverMonitor;
}

template< class Problem >
void tnlExplicitSolver < Problem > :: refreshSolverMonitor()
{
   if( this -> solverMonitor )
   {
      this -> solverMonitor -> setIterations( this -> getIterations() );
      this -> solverMonitor -> setResidue( this -> getResidue() );
      this -> solverMonitor -> setTimeStep( this -> getTau() );
      this -> solverMonitor -> setTime( this -> getTime() );
      this -> solverMonitor -> refresh();
   }
}

template< class Problem >
void tnlExplicitSolver < Problem > :: setTestingMode( bool testingMode )
{
   this -> testingMode = testingMode;
}


#endif
