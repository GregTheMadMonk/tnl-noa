/***************************************************************************
                          tnlMersonSolver_impl.h  -  description
                             -------------------
    begin                : Mar 10, 2013
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

#ifndef tnlMersonSolver_implH
#define tnlMersonSolver_implH

#include <cmath>
#include <core/tnlHost.h>
#include <core/tnlCuda.h>
#include <config/tnlParameterContainer.h>

using namespace std;

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
__global__ void updateU( const Index size,
                         const Real tau,
                         const Real* k1,
                         const Real* k4,
                         const Real* k5,
                         Real* u );
#endif



template< typename Problem >
tnlMersonSolver< Problem > :: tnlMersonSolver()
: k1( "tnlMersonSolver:k1" ),
  k2( "tnlMersonSolver:k2" ),
  k3( "tnlMersonSolver:k3" ),
  k4( "tnlMersonSolver:k4" ),
  k5( "tnlMersonSolver:k5" ),
  kAux( "tnlMersonSolver:kAux" ),
  adaptivity( 0.00001 )
{
   this->setName( "MersonSolver" );
};

template< typename Problem >
tnlString tnlMersonSolver< Problem > :: getType() const
{
   return tnlString( "tnlMersonSolver< " ) +
          Problem :: getTypeStatic() +
          tnlString( " >" );
};

template< typename Problem >
void tnlMersonSolver< Problem > :: configSetup( tnlConfigDescription& config,
                                                const tnlString& prefix )
{
   tnlExplicitSolver< Problem >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "merson-adaptivity", "Time step adaptivity controlling coefficient (the smaller the more precise the computation is, zero means no adaptivity).", 1.0e-4 );
};

template< typename Problem >
bool tnlMersonSolver< Problem > :: init( const tnlParameterContainer& parameters,
                                         const tnlString& prefix )
{
   tnlExplicitSolver< Problem >::init( parameters, prefix );
   if( parameters.CheckParameter( prefix + "merson-adaptivity" ) )
      this->setAdaptivity( parameters.GetParameter< double >( prefix + "merson-adaptivity" ) );
}

template< typename Problem >
void tnlMersonSolver< Problem > :: setAdaptivity( const RealType& a )
{
   this -> adaptivity = a;
};

template< typename Problem >
bool tnlMersonSolver< Problem > :: solve( DofVectorType& u )
{
   if( ! this -> problem )
   {
      cerr << "No problem was set for the Merson ODE solver." << endl;
      return false;
   }
   /****
    * First setup the supporting meshes k1...k5 and kAux.
    */
   if( ! k1. setLike( u ) ||
       ! k2. setLike( u ) ||
       ! k3. setLike( u ) ||
       ! k4. setLike( u ) ||
       ! k5. setLike( u ) ||
       ! kAux. setLike( u ) )
   {
      cerr << "I do not have enough memory to allocate supporting grids for the Merson explicit solver." << endl;
      return false;
   }
   k1. setValue( 0.0 );
   k2. setValue( 0.0 );
   k3. setValue( 0.0 );
   k4. setValue( 0.0 );
   k5. setValue( 0.0 );
   kAux. setValue( 0.0 );


   /****
    * Set necessary parameters
    */
   RealType& time = this -> time;
   RealType currentTau = this -> tau;
   RealType& residue = this -> residue;
   IndexType& iteration = this -> iteration;
   if( time + currentTau > this -> getStopTime() ) currentTau = this -> getStopTime() - time;
   if( currentTau == 0.0 ) return true;
   iteration = 0;

   this -> refreshSolverMonitor();

   /****
    * Start the main loop
    */
   while( 1 )
   {
      /****
       * Compute Runge-Kutta coefficients
       */
      computeKFunctions( u, time, currentTau );
      if( this -> testingMode )
         writeGrids( u );

      /****
       * Compute an error of the approximation.
       */
      RealType eps( 0.0 );
      if( adaptivity != 0.0 )
         eps = computeError( currentTau );

      if( adaptivity == 0.0 || eps < adaptivity )
      {
         RealType lastResidue = residue;
         computeNewTimeLevel( u, currentTau, residue );
         /****
          * When time is close to stopTime the new residue
          * may be inaccurate significantly.
          */
         if( currentTau + time == this -> stopTime ) residue = lastResidue;
         time += currentTau;
         iteration ++;
      }
      this -> refreshSolverMonitor();

      /****
       * Compute the new time step.
       */
      if( adaptivity != 0.0 && eps != 0.0 )
      {
         currentTau *= 0.8 * pow( adaptivity / eps, 0.2 );
         :: MPIBcast( currentTau, 1, 0, this -> solver_comm );
      }
      if( time + currentTau > this -> getStopTime() )
         currentTau = this -> getStopTime() - time; //we don't want to keep such tau
      else this -> tau = currentTau;


      /****
       * Check stop conditions.
       */
      if( time >= this -> getStopTime() ||
          ( this -> getMaxResidue() != 0.0 && residue < this -> getMaxResidue() ) )
       {
         this -> refreshSolverMonitor();
         return true;
       }
      if( iteration == this -> getMaxIterationsNumber() ||
          std::isnan( residue ) )
         return false;
   }
};

template< typename Problem >
void tnlMersonSolver< Problem > :: computeKFunctions( DofVectorType& u,
                                                      const RealType& time,
                                                      RealType tau )
{
   IndexType size = u. getSize();

   RealType* _k1 = k1. getData();
   RealType* _k2 = k2. getData();
   RealType* _k3 = k3. getData();
   RealType* _k4 = k4. getData();
   RealType* _k5 = k5. getData();
   RealType* _kAux = kAux. getData();
   RealType* _u = u. getData();

   /****
    * Compute data transfers statistics
    */
#ifdef HAVE_NOT_CXX11
   k1. template touch< IndexType >( 4 );
   k2. template touch< IndexType >( 1 );
   k3. template touch< IndexType >( 2 );
   k4. template touch< IndexType >( 1 );
   kAux. template touch< IndexType >( 4 );
   u. template touch< IndexType >( 4 );
#else
   k1. touch( 4 );
   k2. touch( 1 );
   k3. touch( 2 );
   k4. touch( 1 );
   kAux. touch( 4 );
   u. touch( 4 );
#endif

   RealType tau_3 = tau / 3.0;

   if( DeviceType :: getDevice() == tnlHostDevice )
   {
      this -> problem -> GetExplicitRHS( time, tau, u, k1 );

   #ifdef HAVE_OPENMP
   #pragma omp parallel for firstprivate( size, _kAux, _u, _k1, tau, tau_3 )
   #endif
      for( IndexType i = 0; i < size; i ++ )
         _kAux[ i ] = _u[ i ] + tau * ( 1.0 / 3.0 * _k1[ i ] );
      this -> problem -> GetExplicitRHS( time + tau_3, tau, kAux, k2 );

   #ifdef HAVE_OPENMP
   #pragma omp parallel for firstprivate( size, _kAux, _u, _k1, _k2, tau, tau_3 )
   #endif
      for( IndexType i = 0; i < size; i ++ )
         _kAux[ i ] = _u[ i ] + tau * 1.0 / 6.0 * ( _k1[ i ] + _k2[ i ] );
      this -> problem -> GetExplicitRHS( time + tau_3, tau, kAux, k3 );

   #ifdef HAVE_OPENMP
   #pragma omp parallel for firstprivate( size, _kAux, _u, _k1, _k3, tau, tau_3 )
   #endif
      for( IndexType i = 0; i < size; i ++ )
         _kAux[ i ] = _u[ i ] + tau * ( 0.125 * _k1[ i ] + 0.375 * _k3[ i ] );
      this -> problem -> GetExplicitRHS( time + 0.5 * tau, tau, kAux, k4 );

   #ifdef HAVE_OPENMP
   #pragma omp parallel for firstprivate( size, _kAux, _u, _k1, _k3, _k4, tau, tau_3 )
   #endif
      for( IndexType i = 0; i < size; i ++ )
         _kAux[ i ] = _u[ i ] + tau * ( 0.5 * _k1[ i ] - 1.5 * _k3[ i ] + 2.0 * _k4[ i ] );
      this -> problem -> GetExplicitRHS( time + tau, tau, kAux, k5 );
   }
   if( DeviceType :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      const int block_size = 512;
      const int grid_size = ( size - 1 ) / block_size + 1;

      this -> problem -> GetExplicitRHS( time, tau, u, k1 );
      cudaThreadSynchronize();

      computeK2Arg<<< grid_size, block_size >>>( size, tau, _u, _k1, _kAux );
      cudaThreadSynchronize();
      this -> problem -> GetExplicitRHS( time + tau_3, tau, kAux, k2 );
      cudaThreadSynchronize();

      computeK3Arg<<< grid_size, block_size >>>( size, tau, _u, _k1, _k2, _kAux );
      cudaThreadSynchronize();
      this -> problem -> GetExplicitRHS( time + tau_3, tau, kAux, k3 );
      cudaThreadSynchronize();

      computeK4Arg<<< grid_size, block_size >>>( size, tau, _u, _k1, _k3, _kAux );
      cudaThreadSynchronize();
      this -> problem -> GetExplicitRHS( time + 0.5 * tau, tau, kAux, k4 );
      cudaThreadSynchronize();

      computeK5Arg<<< grid_size, block_size >>>( size, tau, _u, _k1, _k3, _k4, _kAux );
      cudaThreadSynchronize();
      this -> problem -> GetExplicitRHS( time + tau, tau, kAux, k5 );
      cudaThreadSynchronize();
#endif
   }
}

template< typename Problem >
typename Problem :: RealType tnlMersonSolver< Problem > :: computeError( const RealType tau )
{
   const IndexType size = k1. getSize();
   const RealType* _k1 = k1. getData();
   const RealType* _k3 = k3. getData();
   const RealType* _k4 = k4. getData();
   const RealType* _k5 = k5. getData();
   RealType* _kAux = kAux. getData();

   /****
    * Compute data transfers statistics
    */
#ifdef HAVE_NOT_CXX11
   k1. template touch< IndexType >();
   k3. template touch< IndexType >();
   k4. template touch< IndexType >();
   k5. template touch< IndexType >();
#else
   k1. touch();
   k3. touch();
   k4. touch();
   k5. touch();
#endif

   RealType eps( 0.0 ), maxEps( 0.0 );
   if( DeviceType :: getDevice() == tnlHostDevice )
   {
      // TODO: implement OpenMP support
      for( IndexType i = 0; i < size; i ++  )
      {
         RealType err = ( RealType ) ( tau / 3.0 *
                              fabs( 0.2 * _k1[ i ] +
                                   -0.9 * _k3[ i ] +
                                    0.8 * _k4[ i ] +
                                   -0.1 * _k5[ i ] ) );
         eps = Max( eps, err );
      }
   }
   if( DeviceType :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      const int block_size = 512;
      const int grid_size = ( size - 1 ) / block_size + 1;

      computeErrorKernel<<< grid_size, block_size >>>( size, tau, _k1, _k3, _k4, _k5, _kAux );
      cudaThreadSynchronize();
      eps = tnlMax( kAux );
#endif
   }
   :: MPIAllreduce( eps, maxEps, 1, MPI_MAX, this -> solver_comm );
   return maxEps;
}

template< typename Problem >
void tnlMersonSolver< Problem > :: computeNewTimeLevel( DofVectorType& u,
                                                        RealType tau,
                                                        RealType& currentResidue )
{
   RealType localResidue = RealType( 0.0 );
   IndexType size = k1. getSize();
   RealType* _u = u. getData();
   RealType* _k1 = k1. getData();
   RealType* _k4 = k4. getData();
   RealType* _k5 = k5. getData();

   /****
    * Compute data transfers statistics
    */
#ifdef HAVE_NOT_CXX11
   u. template touch< IndexType >();
   k1. template touch< IndexType >();
   k4. template touch< IndexType >();
   k5. template touch< IndexType >();
#else
   u. touch();
   k1. touch();
   k4. touch();
   k5. touch();
#endif

   if( DeviceType :: getDevice() == tnlHostDevice )
   {
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:localResidue) firstprivate( size, _u, _k1, _k4, _k5, tau )
#endif
      for( IndexType i = 0; i < size; i ++ )
      {
         const RealType add = tau / 6.0 * ( _k1[ i ] + 4.0 * _k4[ i ] + _k5[ i ] );
         _u[ i ] += add;
         localResidue += fabs( ( RealType ) add );
      }
   }
   if( DeviceType :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      const int block_size = 512;
      const int grid_size = ( size - 1 ) / block_size + 1;

      updateU<<< grid_size, block_size >>>( size, tau, _k1, _k4, _k5, _u );
      cudaThreadSynchronize();
      localResidue = 0.0;
#endif
   }
   localResidue /= tau * ( RealType ) size;
   :: MPIAllreduce( localResidue, currentResidue, 1, MPI_SUM, this -> solver_comm );

}

template< typename Problem >
void tnlMersonSolver< Problem > :: writeGrids( const DofVectorType& u )
{
   cout << "Writing Merson solver grids ...";
   u. save( "tnlMersonSolver-u.tnl" );
   k1. save( "tnlMersonSolver-k1.tnl" );
   k2. save( "tnlMersonSolver-k2.tnl" );
   k3. save( "tnlMersonSolver-k3.tnl" );
   k4. save( "tnlMersonSolver-k4.tnl" );
   k5. save( "tnlMersonSolver-k5.tnl" );
   cout << " done. PRESS A KEY." << endl;
   getchar();
}

#ifdef HAVE_CUDA

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
      err[ i ] = 1.0 / 3.0 *  tau * fabs( 0.2 * k1[ i ] +
                                         -0.9 * k3[ i ] +
                                          0.8 * k4[ i ] +
                                         -0.1 * k5[ i ] );
}

template< typename RealType, typename Index >
__global__ void updateU( const Index size,
                         const RealType tau,
                         const RealType* k1,
                         const RealType* k4,
                         const RealType* k5,
                         RealType* u )
{
        Index i = blockIdx. x * blockDim. x + threadIdx. x;
        if( i < size )
                u[ i ] += 1.0 / 6.0 * tau * ( k1[ i ] + 4.0 * k4[ i ] + k5[ i ] );
}

#endif

#endif
