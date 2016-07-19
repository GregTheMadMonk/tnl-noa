/***************************************************************************
                          tnlExplicitSolver_impl.h  -  description
                             -------------------
    begin                : Nov 22, 2014
    copyright            : (C) 2014 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

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
   //tnlIterativeSolver< typename Problem::RealType, typename Problem::IndexType >::configSetup( config, prefix );
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
   if( this->solverMonitor )
   {
      this->solverMonitor->setIterations( this->getIterations() );
      this->solverMonitor->setResidue( this->getResidue() );
      this->solverMonitor->setTimeStep( this->getTau() );
      this->solverMonitor->setTime( this->getTime() );
      this->solverMonitor->setRefreshRate( this->refreshRate );
      this->solverMonitor->refresh();
   }
}

template< class Problem >
void
tnlExplicitSolver< Problem >::
setTestingMode( bool testingMode )
{
   this->testingMode = testingMode;
}

#ifdef HAVE_CUDA
template< typename Real, typename Index >
__device__ void computeBlockResidue( Real* du,
                                     Real* blockResidue,
                                     Index n )
{
   if( n == 128 || n ==  64 || n ==  32 || n ==  16 ||
       n ==   8 || n ==   4 || n ==   2 || n == 256 ||
       n == 512 )
    {
       if( blockDim.x >= 512 )
       {
          if( threadIdx.x < 256 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 256 ];
          __syncthreads();
       }
       if( blockDim.x >= 256 )
       {
          if( threadIdx.x < 128 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 128 ];
          __syncthreads();
       }
       if( blockDim.x >= 128 )
       {
          if( threadIdx.x < 64 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 64 ];
          __syncthreads();
       }

       /***
        * This runs in one warp so it is synchronized implicitly.
        */
       if ( threadIdx.x < 32)
       {
          if( blockDim.x >= 64 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 32 ];
          if( blockDim.x >= 32 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 16 ];
          if( blockDim.x >= 16 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 8 ];
          if( blockDim.x >=  8 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 4 ];
          if( blockDim.x >=  4 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 2 ];
          if( blockDim.x >=  2 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 1 ];
       }
    }
    else
    {
       int s;
       if( n >= 512 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];

          __syncthreads();
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
          __syncthreads();
       }
       if( n >= 256 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];

          __syncthreads();
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
          __syncthreads();
       }
       if( n >= 128 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];

          __syncthreads();
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
          __syncthreads();
       }
       if( n >= 64 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];

          __syncthreads();
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
          __syncthreads();

       }
       if( n >= 32 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];

          __syncthreads();
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
          __syncthreads();
       }
       /***
        * This runs in one warp so it is synchronised implicitly.
        */
       if( n >= 16 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
       }
       if( n >= 8 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
       }
       if( n >= 4 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
       }
       if( n >= 2 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
       }
    }

   if( threadIdx.x == 0 )
      blockResidue[ blockIdx.x ] = du[ 0 ];

}
#endif

} // namespace TNL
