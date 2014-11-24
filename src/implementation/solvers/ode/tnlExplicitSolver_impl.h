/***************************************************************************
                          tnlExplicitSolver_impl.h  -  description
                             -------------------
    begin                : Nov 22, 2014
    copyright            : (C) 2014 by oberhuber
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

#ifndef TNLEXPLICITSOLVER_IMPL_H_
#define TNLEXPLICITSOLVER_IMPL_H_

template< typename Problem >
tnlExplicitSolver< Problem >::
tnlExplicitSolver()
:  time( 0.0 ),
   tau( 0.0 ),
   maxTau( DBL_MAX ),
   stopTime( 0.0 ),
   solver_comm( MPI_COMM_WORLD ),
   verbosity( 0 ),
   cpu_timer( &defaultCPUTimer ),
   rt_timer( &defaultRTTimer ),
   testingMode( false ),
   problem( 0 ),
   solverMonitor( 0 )
{
};

template< typename Problem >
void
tnlExplicitSolver< Problem >::
configSetup( tnlConfigDescription& config,
             const tnlString& prefix )
{
   tnlIterativeSolver< typename Problem::RealType, typename Problem::IndexType >::configSetup( config, prefix );
}

template< typename Problem >
bool
tnlExplicitSolver< Problem >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   return tnlIterativeSolver< typename Problem::RealType, typename Problem::IndexType >::setup( parameters, prefix );
}


template< typename Problem >
void
tnlExplicitSolver< Problem >::
setProblem( Problem& problem )
{
   this->problem = &problem;
};

template< class Problem >
void
tnlExplicitSolver< Problem >::
setTime( const RealType& time )
{
   this->time = time;
};

template< class Problem >
const typename Problem :: RealType&
tnlExplicitSolver< Problem >::
getTime() const
{
   return this->time;
};

template< class Problem >
void
tnlExplicitSolver< Problem >::
setTau( const RealType& tau )
{
   this->tau = tau;
};

template< class Problem >
const typename Problem :: RealType&
tnlExplicitSolver< Problem >::
getTau() const
{
   return this->tau;
};

template< class Problem >
void
tnlExplicitSolver< Problem >::
setMaxTau( const RealType& maxTau )
{
   this->maxTau = maxTau;
};


template< class Problem >
const typename Problem :: RealType&
tnlExplicitSolver< Problem >::
getMaxTau() const
{
   return this->maxTau;
};


template< class Problem >
typename Problem :: RealType
tnlExplicitSolver< Problem >::
getStopTime() const
{
    return this->stopTime;
}

template< class Problem >
void
tnlExplicitSolver< Problem >::
setStopTime( const RealType& stopTime )
{
    this->stopTime = stopTime;
}

template< class Problem >
void
tnlExplicitSolver< Problem >::
setMPIComm( MPI_Comm comm )
{
   this->solver_comm = comm;
};

template< class Problem >
void
tnlExplicitSolver< Problem >::
setVerbose( IndexType v )
{
   this->verbosity = v;
};

template< class Problem >
void
tnlExplicitSolver< Problem >::
setTimerCPU( tnlTimerCPU* timer )
{
   this->cpu_timer = timer;
};

template< class Problem >
void
tnlExplicitSolver< Problem >::
setTimerRT( tnlTimerRT* timer )
{
   this->rt_timer = timer;
};

template< class Problem >
void
tnlExplicitSolver< Problem >::
setSolverMonitor( tnlODESolverMonitor< RealType, IndexType >& solverMonitor )
{
   this->solverMonitor = &solverMonitor;
}

template< class Problem >
void
tnlExplicitSolver< Problem >::
refreshSolverMonitor()
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
void
tnlExplicitSolver< Problem >::
setTestingMode( bool testingMode )
{
   this->testingMode = testingMode;
}

#endif /* TNLEXPLICITSOLVER_IMPL_H_ */
