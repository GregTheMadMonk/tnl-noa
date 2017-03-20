/***************************************************************************
                          Euler_impl.h  -  description
                             -------------------
    begin                : Mar 10, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Solvers {
namespace ODE {

#ifdef HAVE_CUDA
template< typename RealType, typename Index >
__global__ void updateUEuler( const Index size,
                              const RealType tau,
                              const RealType* k1,
                              RealType* u,
                              RealType* cudaBlockResidue );
#endif

template< typename Problem >
Euler< Problem > :: Euler()
: cflCondition( 0.0 )
{
   //timer.reset();
   //updateTimer.reset();
};

template< typename Problem >
String Euler< Problem > :: getType() const
{
   return String( "Euler< " ) +
          Problem :: getTypeStatic() +
          String( " >" );
};

template< typename Problem >
void Euler< Problem > :: configSetup( Config::ConfigDescription& config,
                                               const String& prefix )
{
   //ExplicitSolver< Problem >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "euler-cfl", "Coefficient C in the Courant–Friedrichs–Lewy condition.", 0.0 );
};

template< typename Problem >
bool Euler< Problem > :: setup( const Config::ParameterContainer& parameters,
                                        const String& prefix )
{
   ExplicitSolver< Problem >::setup( parameters, prefix );
   if( parameters.checkParameter( prefix + "euler-cfl" ) )
      this->setCFLCondition( parameters.getParameter< double >( prefix + "euler-cfl" ) );
   return true;
}

template< typename Problem >
void Euler< Problem > :: setCFLCondition( const RealType& cfl )
{
   this->cflCondition = cfl;
}

template< typename Problem >
const typename Problem :: RealType& Euler< Problem > :: getCFLCondition() const
{
   return this->cflCondition;
}

template< typename Problem >
bool Euler< Problem > :: solve( DofVectorPointer& u )
{
   /****
    * First setup the supporting meshes k1...k5 and k_tmp.
    */
   //timer.start();
   if( ! k1->setLike( *u ) )
   {
      std::cerr << "I do not have enough memory to allocate a supporting grid for the Euler explicit solver." << std::endl;
      return false;
   }
   k1->setValue( 0.0 );


   /****
    * Set necessary parameters
    */
   RealType& time = this->time;
   RealType currentTau = min( this->getTau(), this->getMaxTau() );
   if( time + currentTau > this->getStopTime() ) currentTau = this->getStopTime() - time;
   if( currentTau == 0.0 ) return true;
   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   this->refreshSolverMonitor();

   /****
    * Start the main loop
    */
   while( 1 )
   {
      /****
       * Compute the RHS
       */
      //timer.stop();
      this->problem->getExplicitRHS( time, currentTau, u, k1 );
      //timer.start();

      RealType lastResidue = this->getResidue();
      RealType maxResidue( 0.0 );
      if( this->cflCondition != 0.0 )
      {
         maxResidue = k1->absMax();
         if( currentTau * maxResidue > this->cflCondition )
         {
            currentTau *= 0.9;
            continue;
         }
      }
      RealType newResidue( 0.0 );
      //updateTimer.start();
      computeNewTimeLevel( u, currentTau, newResidue );
      //updateTimer.stop();
      this->setResidue( newResidue );

      /****
       * When time is close to stopTime the new residue
       * may be inaccurate significantly.
       */
      if( currentTau + time == this->stopTime ) this->setResidue( lastResidue );
      time += currentTau;

      if( ! this->nextIteration() )
         return this->checkConvergence();

      /****
       * Compute the new time step.
       */
      if( time + currentTau > this->getStopTime() )
         currentTau = this->getStopTime() - time; //we don't want to keep such tau
      else this->tau = currentTau;

      this->refreshSolverMonitor();

      /****
       * Check stop conditions.
       */
      if( time >= this->getStopTime() ||
          ( this->getConvergenceResidue() != 0.0 && this->getResidue() < this->getConvergenceResidue() ) )
      {
         this->refreshSolverMonitor();
         //std::cerr << std::endl << "RHS Timer = " << timer.getRealTime() << std::endl;
         //std::cerr << std::endl << "Update Timer = " << updateTimer.getRealTime() << std::endl;
         return true;
      }

      if( this->cflCondition != 0.0 )
      {
         currentTau /= 0.95;
         currentTau = min( currentTau, this->getMaxTau() );
      }
   }
};

template< typename Problem >
void Euler< Problem > :: computeNewTimeLevel( DofVectorPointer& u,
                                                       RealType tau,
                                                       RealType& currentResidue )
{
   RealType localResidue = RealType( 0.0 );
   const IndexType size = k1->getSize();
   RealType* _u = u->getData();
   RealType* _k1 = k1->getData();

   if( std::is_same< DeviceType, Devices::Host >::value )
   {
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:localResidue) firstprivate( _u, _k1, tau ) if( Devices::Host::isOMPEnabled() )
#endif
      for( IndexType i = 0; i < size; i ++ )
      {
         const RealType add = tau * _k1[ i ];
         _u[ i ] += add;
         localResidue += std::fabs( add );
      }
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

         updateUEuler<<< cudaBlocks, cudaBlockSize, sharedMemory >>>( currentSize,
                                                                      tau,
                                                                      &_k1[ gridOffset ],
                                                                      &_u[ gridOffset ],
                                                                      this->cudaBlockResidue.getData() );
         localResidue += this->cudaBlockResidue.sum();
         cudaThreadSynchronize();
         checkCudaDevice;
      }
#endif
   }
   localResidue /= tau * ( RealType ) size;
   MPIAllreduce( localResidue, currentResidue, 1, MPI_SUM, this->solver_comm );
}

#ifdef HAVE_CUDA
template< typename RealType, typename Index >
__global__ void updateUEuler( const Index size,
                              const RealType tau,
                              const RealType* k1,
                              RealType* u,
                              RealType* cudaBlockResidue )
{
   extern __shared__ RealType du[];
   const Index blockOffset = blockIdx. x * blockDim. x;
   const Index i = blockOffset  + threadIdx. x;
   if( i < size )
      u[ i ] += du[ threadIdx.x ] = tau * k1[ i ];
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
