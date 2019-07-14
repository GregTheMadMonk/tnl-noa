/***************************************************************************
                          ExplicitSolver_impl.h  -  description
                             -------------------
    begin                : Nov 22, 2014
    copyright            : (C) 2014 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Solvers {
namespace ODE {   

template< typename Problem, typename SolverMonitor >
ExplicitSolver< Problem, SolverMonitor >::
ExplicitSolver()
:  time( 0.0 ),
   stopTime( 0.0 ),
   tau( 0.0 ),
   maxTau( DBL_MAX ),
   verbosity( 0 ),
   testingMode( false ),
   problem( 0 )//,
   //solverMonitor( 0 )
{
};

template< typename Problem, typename SolverMonitor >
void
ExplicitSolver< Problem, SolverMonitor >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   //IterativeSolver< typename Problem::RealType, typename Problem::IndexType >::configSetup( config, prefix );
}

template< typename Problem, typename SolverMonitor >
bool
ExplicitSolver< Problem, SolverMonitor >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   this->setVerbose( parameters.getParameter< int >( "verbose" ) );
   return IterativeSolver< typename Problem::RealType, typename Problem::IndexType, SolverMonitor >::setup( parameters, prefix );
}

template< typename Problem, typename SolverMonitor >
void
ExplicitSolver< Problem, SolverMonitor >::
setProblem( Problem& problem )
{
   this->problem = &problem;
};

template< class Problem, typename SolverMonitor >
void
ExplicitSolver< Problem, SolverMonitor >::
setTime( const RealType& time )
{
   this->time = time;
};

template< class Problem, typename SolverMonitor >
const typename Problem :: RealType&
ExplicitSolver< Problem, SolverMonitor >::
getTime() const
{
   return this->time;
};

template< class Problem, typename SolverMonitor >
void
ExplicitSolver< Problem, SolverMonitor >::
setTau( const RealType& tau )
{
   this->tau = tau;
};

template< class Problem, typename SolverMonitor >
const typename Problem :: RealType&
ExplicitSolver< Problem, SolverMonitor >::
getTau() const
{
   return this->tau;
};

template< class Problem, typename SolverMonitor >
void
ExplicitSolver< Problem, SolverMonitor >::
setMaxTau( const RealType& maxTau )
{
   this->maxTau = maxTau;
};


template< class Problem, typename SolverMonitor >
const typename Problem :: RealType&
ExplicitSolver< Problem, SolverMonitor >::
getMaxTau() const
{
   return this->maxTau;
};


template< class Problem, typename SolverMonitor >
typename Problem :: RealType
ExplicitSolver< Problem, SolverMonitor >::
getStopTime() const
{
    return this->stopTime;
}

template< class Problem, typename SolverMonitor >
void
ExplicitSolver< Problem, SolverMonitor >::
setStopTime( const RealType& stopTime )
{
    this->stopTime = stopTime;
}

template< class Problem, typename SolverMonitor >
void
ExplicitSolver< Problem, SolverMonitor >::
setVerbose( IndexType v )
{
   this->verbosity = v;
};

template< class Problem, typename SolverMonitor >
void
ExplicitSolver< Problem, SolverMonitor >::
refreshSolverMonitor( bool force )
{
   if( this->solverMonitor )
   {
      this->solverMonitor->setIterations( this->getIterations() );
      this->solverMonitor->setResidue( this->getResidue() );
      this->solverMonitor->setTimeStep( this->getTau() );
      this->solverMonitor->setTime( this->getTime() );
      this->solverMonitor->setRefreshRate( this->refreshRate );
   }
}

template< class Problem, typename SolverMonitor >
void
ExplicitSolver< Problem, SolverMonitor >::
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

} // namespace ODE
} // namespace Solvers
} // namespace TNL
