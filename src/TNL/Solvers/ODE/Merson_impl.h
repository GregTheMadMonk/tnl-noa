/***************************************************************************
                          Merson_impl.h  -  description
                             -------------------
    begin                : Mar 10, 2013
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Communicators/NoDistrCommunicator.h>

#include "Merson.h"

namespace TNL {
namespace Solvers {
namespace ODE {   

/****
 * In this code we do not use constants and references as we would like to.
 * OpenMP would complain that
 *
 *  error: ‘some-variable’ is predetermined ‘shared’ for ‘firstprivate’
 *
 */

#ifdef HAVE_CUDA

template< typename Real, typename Index >
__global__ void computeK2Arg( const Index size,
                              const Real tau,
                              const Real* u,
                              const Real* k1,
                              Real* k2_arg );

template< typename Real, typename Index >
__global__ void computeK3Arg( const Index size,
                              const Real tau,
                              const Real* u,
                              const Real* k1,
                              const Real* k2,
                              Real* k3_arg );

template< typename Real, typename Index >
__global__ void computeK4Arg( const Index size,
                              const Real tau,
                              const Real* u,
                              const Real* k1,
                              const Real* k3,
                              Real* k4_arg );

template< typename Real, typename Index >
__global__ void computeK5Arg( const Index size,
                              const Real tau,
                              const Real* u,
                              const Real* k1,
                              const Real* k3,
                              const Real* k4,
                              Real* k5_arg );

template< typename Real, typename Index >
__global__ void computeErrorKernel( const Index size,
                                    const Real tau,
                                    const Real* k1,
                                    const Real* k3,
                                    const Real* k4,
                                    const Real* k5,
                                    Real* err );

template< typename Real, typename Index >
__global__ void updateUMerson( const Index size,
                               const Real tau,
                               const Real* k1,
                               const Real* k4,
                               const Real* k5,
                               Real* u,
                               Real* blockResidue );
#endif



template< typename Problem >
Merson< Problem > :: Merson()
: adaptivity( 0.00001 )
{
   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      this->openMPErrorEstimateBuffer.setSize( std::max( 1, Devices::Host::getMaxThreadsCount() ) );
   }
};

template< typename Problem >
String Merson< Problem > :: getType()
{
   return String( "Merson< " ) +
          Problem::getType() +
          String( " >" );
};

template< typename Problem >
void Merson< Problem > :: configSetup( Config::ConfigDescription& config,
                                                const String& prefix )
{
   //ExplicitSolver< Problem >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "merson-adaptivity", "Time step adaptivity controlling coefficient (the smaller the more precise the computation is, zero means no adaptivity).", 1.0e-4 );
};

template< typename Problem >
bool Merson< Problem > :: setup( const Config::ParameterContainer& parameters,
                                         const String& prefix )
{
   ExplicitSolver< Problem >::setup( parameters, prefix );
   if( parameters.checkParameter( prefix + "merson-adaptivity" ) )
      this->setAdaptivity( parameters.getParameter< double >( prefix + "merson-adaptivity" ) );
   return true;
}

template< typename Problem >
void Merson< Problem > :: setAdaptivity( const RealType& a )
{
   this->adaptivity = a;
};

template< typename Problem >
bool Merson< Problem >::solve( DofVectorPointer& _u )
{
   if( ! this->problem )
   {
      std::cerr << "No problem was set for the Merson ODE solver." << std::endl;
      return false;
   }
   if( this->getTau() == 0.0 )
   {
      std::cerr << "The time step for the Merson ODE solver is zero." << std::endl;
      return false;
   }
   /****
    * First setup the supporting meshes k1...k5 and kAux.
    */
   _k1->setLike( *_u );
   _k2->setLike( *_u );
   _k3->setLike( *_u );
   _k4->setLike( *_u );
   _k5->setLike( *_u );
   _kAux->setLike( *_u );
   auto k1 = _k1->getView();
   auto k2 = _k2->getView();
   auto k3 = _k3->getView();
   auto k4 = _k4->getView();
   auto k5 = _k5->getView();
   auto kAux = _kAux->getView();
   auto u = _u->getView();
   k1 = 0.0;
   k2 = 0.0;
   k3 = 0.0;
   k4 = 0.0;
   k5 = 0.0;
   kAux = 0.0;

   /****
    * Set necessary parameters
    */
   RealType& time = this->time;
   RealType currentTau = min( this->getTau(), this->getMaxTau() );
   if( time + currentTau > this->getStopTime() )
      currentTau = this->getStopTime() - time;
   if( currentTau == 0.0 ) return true;
   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   this->refreshSolverMonitor();

   /////
   // Start the main loop
   while( this->checkNextIteration() )
   {
      /////
      // Compute Runge-Kutta coefficients
      RealType tau_3 = currentTau / 3.0;

      /////
      // k1
      this->problem->getExplicitUpdate( time, currentTau, _u, _k1 );

      /////
      // k2
      kAux = u + currentTau * ( 1.0 / 3.0 * k1 );
      this->problem->applyBoundaryConditions( time + tau_3, _kAux );
      this->problem->getExplicitUpdate( time + tau_3, currentTau, _kAux, _k2 );

      /////
      // k3
      kAux = u + currentTau * 1.0 / 6.0 * ( k1 + k2 );
      this->problem->applyBoundaryConditions( time + tau_3, _kAux );
      this->problem->getExplicitUpdate( time + tau_3, currentTau, _kAux, _k3 );

      /////
      // k4
      kAux = u + currentTau * ( 0.125 * k1 + 0.375 * k3 );
      this->problem->applyBoundaryConditions( time + 0.5 * currentTau, _kAux );
      this->problem->getExplicitUpdate( time + 0.5 * currentTau, currentTau, _kAux, _k4 );

      /////
      // k5
      kAux = u + currentTau * ( 0.5 * k1 - 1.5 * k3 + 2.0 * k4 );
      this->problem->applyBoundaryConditions( time + currentTau, _kAux );
      this->problem->getExplicitUpdate( time + currentTau, currentTau, _kAux, _k5 );

      if( this->testingMode )
         writeGrids( _u );

      /////
      // Compute an error of the approximation.
      RealType error( 0.0 );
      if( adaptivity != 0.0 )
      {
         const RealType localError = 
            max( currentTau / 3.0 * abs( 0.2 * k1 -0.9 * k3 + 0.8 * k4 -0.1 * k5 ) );
            Problem::CommunicatorType::Allreduce( &localError, &error, 1, MPI_MAX, Problem::CommunicatorType::AllGroup );
      }

      if( adaptivity == 0.0 || error < adaptivity )
      {
         RealType lastResidue = this->getResidue();
         RealType newResidue( 0.0 );
         time += currentTau;

         auto reduction = [] __cuda_callable__ ( RealType& a , const RealType& b ) { a += b; };
         auto volatileReduction = [] __cuda_callable__ ( volatile RealType& a , const volatile RealType& b ) { a += b; };
         this->setResidue( addAndReduceAbs( u, currentTau / 6.0 * ( k1 + 4.0 * k4 + k5 ),
            reduction, volatileReduction, ( RealType ) 0.0 ) / ( currentTau * ( RealType ) u.getSize() ) );

         /////
         // When time is close to stopTime the new residue
         // may be inaccurate significantly.
         if( abs( time - this->stopTime ) < 1.0e-7 ) this->setResidue( lastResidue );
         

         if( ! this->nextIteration() )
            return false;
      }
      this->refreshSolverMonitor();

      /////
      // Compute the new time step.
      if( adaptivity != 0.0 && error != 0.0 )
      {
         currentTau *= 0.8 * ::pow( adaptivity / error, 0.2 );
         currentTau = min( currentTau, this->getMaxTau() );
#ifdef USE_MPI
         TNLMPI::Bcast( currentTau, 1, 0 );
#endif        
      }
      if( time + currentTau > this->getStopTime() )
         currentTau = this->getStopTime() - time; //we don't want to keep such tau
      else this->tau = currentTau;


      /////
      // Check stop conditions.
      if( time >= this->getStopTime() ||
          ( this->getConvergenceResidue() != 0.0 && this->getResidue() < this->getConvergenceResidue() ) )
      {
         this->refreshSolverMonitor( true );
         return true;
      }
   }
   this->refreshSolverMonitor( true );
   return this->checkConvergence();

};

template< typename Problem >
void Merson< Problem >::computeKFunctions( DofVectorPointer& _u,
                                             const RealType& time,
                                             RealType tau )
{
   auto k1 = _k1->getView();
   auto k2 = _k2->getView();
   auto k3 = _k3->getView();
   auto k4 = _k4->getView();
   auto k5 = _k5->getView();
   auto kAux = _kAux->getView();
   auto u = _u->getView();

   RealType tau_3 = tau / 3.0;
   /////
   // k1
   this->problem->getExplicitUpdate( time, tau, _u, _k1 );

   /////
   // k2
   kAux = u + tau * ( 1.0 / 3.0 * k1 );
   this->problem->applyBoundaryConditions( time + tau_3, _kAux );
   this->problem->getExplicitUpdate( time + tau_3, tau, _kAux, _k2 );

   /////
   // k3
   kAux = u + tau * 1.0 / 6.0 * ( k1 + k2 );
   this->problem->applyBoundaryConditions( time + tau_3, _kAux );
   this->problem->getExplicitUpdate( time + tau_3, tau, _kAux, _k3 );

   /////
   // k4
   kAux = u + tau * ( 0.125 * k1 + 0.375 * k3 );
   this->problem->applyBoundaryConditions( time + 0.5 * tau, _kAux );
   this->problem->getExplicitUpdate( time + 0.5 * tau, tau, _kAux, _k4 );

   /////
   // k5
   kAux = u + tau * ( 0.5 * k1 - 1.5 * k3 + 2.0 * k4 );
   this->problem->applyBoundaryConditions( time + tau, _kAux );
   this->problem->getExplicitUpdate( time + tau, tau, _kAux, _k5 );
}

/*template< typename Problem >
typename Problem :: RealType Merson< Problem > :: computeError( const RealType tau )
{
   const IndexType size = _k1->getSize();
   const RealType* k1 = _k1->getData();
   const RealType* k3 = _k3->getData();
   const RealType* k4 = _k4->getData();
   const RealType* k5 = _k5->getData();
   RealType* kAux = _kAux->getData();

   RealType eps( 0.0 ), maxEps( 0.0 );
   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      this->openMPErrorEstimateBuffer.setValue( 0.0 );
#ifdef HAVE_OPENMP
#pragma omp parallel if( Devices::Host::isOMPEnabled() )
#endif
      {
         RealType localEps( 0.0 );
#ifdef HAVE_OPENMP
#pragma omp for
#endif
         for( IndexType i = 0; i < size; i ++  )
         {
            RealType err = ( RealType ) ( tau / 3.0 *
                                 abs( 0.2 * k1[ i ] +
                                     -0.9 * k3[ i ] +
                                      0.8 * k4[ i ] +
                                     -0.1 * k5[ i ] ) );
            localEps = max( localEps, err );
         }
         this->openMPErrorEstimateBuffer[ Devices::Host::getThreadIdx() ] = localEps;
      }
      eps = this->openMPErrorEstimateBuffer.max();
   }
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      dim3 cudaBlockSize( 512 );
      const IndexType cudaBlocks = Devices::Cuda::getNumberOfBlocks( size, cudaBlockSize.x );
      const IndexType cudaGrids = Devices::Cuda::getNumberOfGrids( cudaBlocks );
      this->cudaBlockResidue.setSize( min( cudaBlocks, Devices::Cuda::getMaxGridSize() ) );
      const IndexType threadsPerGrid = Devices::Cuda::getMaxGridSize() * cudaBlockSize.x;

      for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx ++ )
      {
         const IndexType gridOffset = gridIdx * threadsPerGrid;
         const IndexType currentSize = min( size - gridOffset, threadsPerGrid );
         computeErrorKernel<<< cudaBlocks, cudaBlockSize >>>( currentSize,
                                                              tau,
                                                              &k1[ gridOffset ],
                                                              &k3[ gridOffset ],
                                                              &k4[ gridOffset ],
                                                              &k5[ gridOffset ],
                                                              &kAux[ gridOffset ] );
         cudaDeviceSynchronize();
         eps = std::max( eps, _kAux->max() );
      }
#endif
   }
   Problem::CommunicatorType::Allreduce( &eps, &maxEps, 1, MPI_MAX, Problem::CommunicatorType::AllGroup );
   return maxEps;
}

template< typename Problem >
void Merson< Problem >::computeNewTimeLevel( const RealType time,
                                             const RealType tau,
                                             DofVectorPointer& u,
                                             RealType& currentResidue )
{
   RealType localResidue = RealType( 0.0 );
   IndexType size = _k1->getSize();
   RealType* _u = u->getData();
   RealType* k1 = _k1->getData();
   RealType* k4 = _k4->getData();
   RealType* k5 = _k5->getData();

   if( std::is_same< DeviceType, Devices::Host >::value )
   {
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:localResidue) firstprivate( size, _u, k1, k4, k5, tau ) if( Devices::Host::isOMPEnabled() )
#endif
      for( IndexType i = 0; i < size; i ++ )
      {
         const RealType add = tau / 6.0 * ( k1[ i ] + 4.0 * k4[ i ] + k5[ i ] );
         _u[ i ] += add;
         localResidue += abs( ( RealType ) add );
      }
      this->problem->applyBoundaryConditions( time, u );
   }
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      dim3 cudaBlockSize( 512 );
      const IndexType cudaBlocks = Devices::Cuda::getNumberOfBlocks( size, cudaBlockSize.x );
      const IndexType cudaGrids = Devices::Cuda::getNumberOfGrids( cudaBlocks );
      this->cudaBlockResidue.setSize( min( cudaBlocks, Devices::Cuda::getMaxGridSize() ) );
      const IndexType threadsPerGrid = Devices::Cuda::getMaxGridSize() * cudaBlockSize.x;

      localResidue = 0.0;
      for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx ++ )
      {
         const IndexType sharedMemory = cudaBlockSize.x * sizeof( RealType );
         const IndexType gridOffset = gridIdx * threadsPerGrid;
         const IndexType currentSize = min( size - gridOffset, threadsPerGrid );

         updateUMerson<<< cudaBlocks, cudaBlockSize, sharedMemory >>>( currentSize,
                                                                       tau,
                                                                       &k1[ gridOffset ],
                                                                       &k4[ gridOffset ],
                                                                       &k5[ gridOffset ],
                                                                       &_u[ gridOffset ],
                                                                       this->cudaBlockResidue.getData() );
         localResidue += this->cudaBlockResidue.sum();
         cudaDeviceSynchronize();
      }
      this->problem->applyBoundaryConditions( time, u );

#endif
   }

   localResidue /= tau * ( RealType ) size;
   Problem::CommunicatorType::Allreduce( &localResidue, &currentResidue, 1, MPI_SUM, Problem::CommunicatorType::AllGroup);
/*#ifdef USE_MPI
   TNLMPI::Allreduce( localResidue, currentResidue, 1, MPI_SUM);
#else
   currentResidue=localResidue;
#endif*/

}

template< typename Problem >
void Merson< Problem >::writeGrids( const DofVectorPointer& u )
{
   std::cout << "Writing Merson solver grids ...";
   File( "Merson-u.tnl", std::ios_base::out ) << *u;
   File( "Merson-k1.tnl", std::ios_base::out ) << *_k1;
   File( "Merson-k2.tnl", std::ios_base::out ) << *_k2;
   File( "Merson-k3.tnl", std::ios_base::out ) << *_k3;
   File( "Merson-k4.tnl", std::ios_base::out ) << *_k4;
   File( "Merson-k5.tnl", std::ios_base::out ) << *_k5;
   std::cout << " done. PRESS A KEY." << std::endl;
   getchar();
}

#ifdef HAVE_CUDA
/*
template< typename RealType, typename Index >
__global__ void computeK2Arg( const Index size,
                              const RealType tau,
                              const RealType* u,
                              const RealType* k1,
                              RealType* k2_arg )
{
   int i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      k2_arg[ i ] = u[ i ] + tau * ( 1.0 / 3.0 * k1[ i ] );
}

template< typename RealType, typename Index >
__global__ void computeK3Arg( const Index size,
                              const RealType tau,
                              const RealType* u,
                              const RealType* k1,
                              const RealType* k2,
                              RealType* k3_arg )
{
   Index i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      k3_arg[ i ] = u[ i ] + tau * 1.0 / 6.0 * ( k1[ i ] + k2[ i ] );
}

template< typename RealType, typename Index >
__global__ void computeK4Arg( const Index size,
                              const RealType tau,
                              const RealType* u,
                              const RealType* k1,
                              const RealType* k3,
                              RealType* k4_arg )
{
   Index i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      k4_arg[ i ] = u[ i ] + tau * ( 0.125 * k1[ i ] + 0.375 * k3[ i ] );
}

template< typename RealType, typename Index >
__global__ void computeK5Arg( const Index size,
                              const RealType tau,
                              const RealType* u,
                              const RealType* k1,
                              const RealType* k3,
                              const RealType* k4,
                              RealType* k5_arg )
{
   Index i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      k5_arg[ i ] = u[ i ] + tau * ( 0.5 * k1[ i ] - 1.5 * k3[ i ] + 2.0 * k4[ i ] );
}
*/

template< typename RealType, typename Index >
__global__ void computeErrorKernel( const Index size,
                                    const RealType tau,
                                    const RealType* k1,
                                    const RealType* k3,
                                    const RealType* k4,
                                    const RealType* k5,
                                    RealType* err )
{
   Index i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      err[ i ] = 1.0 / 3.0 *  tau * abs( 0.2 * k1[ i ] +
                                        -0.9 * k3[ i ] +
                                         0.8 * k4[ i ] +
                                        -0.1 * k5[ i ] );
}

template< typename RealType, typename Index >
__global__ void updateUMerson( const Index size,
                               const RealType tau,
                               const RealType* k1,
                               const RealType* k4,
                               const RealType* k5,
                               RealType* u,
                               RealType* cudaBlockResidue )
{
   extern __shared__ RealType du[];
   const Index blockOffset = blockIdx. x * blockDim. x;
   const Index i = blockOffset  + threadIdx. x;
   if( i < size )
      u[ i ] += du[ threadIdx.x ] = 1.0 / 6.0 * tau * ( k1[ i ] + 4.0 * k4[ i ] + k5[ i ] );
   else
      du[ threadIdx.x ] = 0.0;
   du[ threadIdx.x ] = abs( du[ threadIdx.x ] );
   __syncthreads();

   const Index rest = size - blockOffset;
   Index n =  rest < blockDim.x ? rest : blockDim.x;

   computeBlockResidue( du,
                        cudaBlockResidue,
                        n );
}

#endif

} // namespace ODE
} // namespace Solvers
} // namespace TNL
