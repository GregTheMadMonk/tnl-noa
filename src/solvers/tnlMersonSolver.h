/***************************************************************************
                          tnlMersonSolver.h  -  description
                             -------------------
    begin                : 2007/06/16
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

#ifndef tnlMersonSolverH
#define tnlMersonSolverH

#include <math.h>
#include <core/low-level/cuda-long-vector-kernels.h>
#include <solvers/tnlExplicitSolver.h>

/****
 * In this code we do not use constants and references as we would like to.
 * OpenMP would complain that
 *
 *  error: ‘some-variable’ is predetermined ‘shared’ for ‘firstprivate’
 *
 */

template< class Problem, class Mesh, typename Real = double, typename Device = tnlHost, typename Index = int >
class tnlMersonSolver : public tnlExplicitSolver< Problem, Mesh, Real, Device, Index >
{
   public:

   tnlMersonSolver( const tnlString& name );

   tnlString getType() const;

   void setAdaptivity( const Real& a );
   
   bool solve( Problem& problem,
               Mesh& u );

   protected:
   
   //! Compute the Runge-Kutta coefficients
   /****
    * The parameter u is not constant because one often
    * needs to correct u on the boundaries to be able to compute
    * the RHS.
    */
   void computeKFunctions( Mesh& u,
                           Problem& problem,
                           const Real& time,
                           Real tau );

   Real computeError( const Real tau );

   void computeNewTimeLevel( Mesh& u,
                             Real tau,
                             Real& currentResidue );

   void writeGrids( const Mesh& u );

   Mesh k1, k2, k3, k4, k5, kAux;

   //! This controls the accuracy of the solver
   Real adaptivity;
};

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



template< class Problem, class Mesh, typename Real, typename Device, typename Index >
tnlMersonSolver< Problem, Mesh, Real, Device, Index > :: tnlMersonSolver( const tnlString& name )
: tnlExplicitSolver< Problem, Mesh, Real, Device, Index >( name ),
  k1( "tnlMersonSolver:k1" ),
  k2( "tnlMersonSolver:k2" ),
  k3( "tnlMersonSolver:k3" ),
  k4( "tnlMersonSolver:k4" ),
  k5( "tnlMersonSolver:k5" ),
  kAux( "tnlMersonSolver:kAux" ),
  adaptivity( 0.00001 )
{
   this -> tau = 1.0;
};

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
tnlString tnlMersonSolver< Problem, Mesh, Real, Device, Index > :: getType() const
{
   Mesh m( "m" );
   Problem p( "p" );
   return tnlString( "tnlMersonSolver< " ) +
          p. getType() +
          tnlString( ", " ) +
          m. getType() +
          tnlString( ", " ) +
          GetParameterType( Real ( 0  ) ) +
          tnlString( ", " ) +
          Device :: getDeviceType() +
          tnlString( ", " ) +
          tnlString( GetParameterType( Index ( 0 ) ) ) +
          tnlString( ", " ) +
          tnlString( " >" );
};

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
void tnlMersonSolver< Problem, Mesh, Real, Device, Index > :: setAdaptivity( const Real& a )
{
   adaptivity = a;
};

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
bool tnlMersonSolver< Problem, Mesh, Real, Device, Index > :: solve( Problem& problem,
                                                                     Mesh& u )
{
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
   Real& time = this -> time;
   Real currentTau = this -> tau;
   Real& residue = this -> residue;
   Index& iteration = this -> iteration;
   if( time + currentTau > this -> getStopTime() ) currentTau = this -> getStopTime() - time;
   if( currentTau == 0.0 ) return true;
   iteration = 0;

   /****
    * Do a printout ...
    */
   if( this -> verbosity > 0 )
      this -> printOut();

   /****
    * Start the main loop
    */
   while( 1 )
   {
      /****
       * Compute Runge-Kutta coefficients
       */
      computeKFunctions( u, problem, time, currentTau );
      if( this -> testingMode )
         writeGrids( u );

      /****
       * Compute an error of the approximation.
       */
      Real eps( 0.0 );
      if( adaptivity != 0.0 )
         eps = computeError( currentTau );

      if( adaptivity == 0.0 || eps < adaptivity )
      {
         Real lastResidue = residue;
         computeNewTimeLevel( u, currentTau, residue );
         /****
          * When time is close to stopTime the new residue
          * may be inaccurate significantly.
          */
         if( currentTau + time == this -> stopTime ) residue = lastResidue;
         time += currentTau;
         iteration ++;
      }

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
       * Do printouts if verbosity is on
       */
      if( this -> verbosity > 1 )
          this ->  printOut();

      /****
       * Check stop conditions.
       */
      if( time >= this -> getStopTime() ||
          ( this -> getMaxResidue() != 0.0 && residue < this -> getMaxResidue() ) )
       {
         if( this -> verbosity > 0 )
            this -> printOut();
          return true;
       }
      if( iteration == this -> getMaxIterationsNumber() ) return false;
   }
};

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
void tnlMersonSolver< Problem, Mesh, Real, Device, Index > :: computeKFunctions( Mesh& u,
                                                                                 Problem& problem,
                                                                                 const Real& time,
                                                                                 Real tau )
{
   Index size = u. getSize();

   Real* _k1 = k1. getData();
   Real* _k2 = k2. getData();
   Real* _k3 = k3. getData();
   Real* _k4 = k4. getData();
   Real* _k5 = k5. getData();
   Real* _kAux = kAux. getData();
   Real* _u = u. getData();

   /****
    * Compute data transfers statistics
    */
   k1. touch( 4 );
   k2. touch( 1 );
   k3. touch( 2 );
   k4. touch( 1 );
   kAux. touch( 4 );
   u. touch( 4 );

   Real tau_3 = tau / 3.0;

   if( Device :: getDevice() == tnlHostDevice )
   {
      problem. GetExplicitRHS( time, u, k1 );

   #ifdef HAVE_OPENMP
   #pragma omp parallel for firstprivate( size, _kAux, _u, _k1, tau, tau_3 )
   #endif
      for( Index i = 0; i < size; i ++ )
         _kAux[ i ] = _u[ i ] + tau * ( 1.0 / 3.0 * _k1[ i ] );
      problem. GetExplicitRHS( time + tau_3, kAux, k2 );

   #ifdef HAVE_OPENMP
   #pragma omp parallel for firstprivate( size, _kAux, _u, _k1, _k2, tau, tau_3 )
   #endif
      for( Index i = 0; i < size; i ++ )
         _kAux[ i ] = _u[ i ] + tau * 1.0 / 6.0 * ( _k1[ i ] + _k2[ i ] );
      problem. GetExplicitRHS( time + tau_3, kAux, k3 );

   #ifdef HAVE_OPENMP
   #pragma omp parallel for firstprivate( size, _kAux, _u, _k1, _k3, tau, tau_3 )
   #endif
      for( Index i = 0; i < size; i ++ )
         _kAux[ i ] = _u[ i ] + tau * ( 0.125 * _k1[ i ] + 0.375 * _k3[ i ] );
      problem. GetExplicitRHS( time + 0.5 * tau, kAux, k4 );

   #ifdef HAVE_OPENMP
   #pragma omp parallel for firstprivate( size, _kAux, _u, _k1, _k3, _k4, tau, tau_3 )
   #endif
      for( Index i = 0; i < size; i ++ )
         _kAux[ i ] = _u[ i ] + tau * ( 0.5 * _k1[ i ] - 1.5 * _k3[ i ] + 2.0 * _k4[ i ] );
      problem. GetExplicitRHS( time + tau, kAux, k5 );
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      const int block_size = 512;
      const int grid_size = ( size - 1 ) / block_size + 1;

      problem. GetExplicitRHS( time, u, k1 );
      cudaThreadSynchronize();

      computeK2Arg<<< grid_size, block_size >>>( size, tau, _u, _k1, _kAux );
      cudaThreadSynchronize();
      problem. GetExplicitRHS( time + tau_3, kAux, k2 );
      cudaThreadSynchronize();

      computeK3Arg<<< grid_size, block_size >>>( size, tau, _u, _k1, _k2, _kAux );
      cudaThreadSynchronize();
      problem. GetExplicitRHS( time + tau_3, kAux, k3 );
      cudaThreadSynchronize();

      computeK4Arg<<< grid_size, block_size >>>( size, tau, _u, _k1, _k3, _kAux );
      cudaThreadSynchronize();
      problem. GetExplicitRHS( time + 0.5 * tau, kAux, k4 );
      cudaThreadSynchronize();

      computeK5Arg<<< grid_size, block_size >>>( size, tau, _u, _k1, _k3, _k4, _kAux );
      cudaThreadSynchronize();
      problem. GetExplicitRHS( time + tau, kAux, k5 );
      cudaThreadSynchronize();
#endif
   }
}

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
Real tnlMersonSolver< Problem, Mesh, Real, Device, Index > :: computeError( const Real tau )
{
   const Index size = k1. getSize();
   const Real* _k1 = k1. getData();
   const Real* _k3 = k3. getData();
   const Real* _k4 = k4. getData();
   const Real* _k5 = k5. getData();
   Real* _kAux = kAux. getData();

   /****
    * Compute data transfers statistics
    */
   k1. touch();
   k3. touch();
   k4. touch();
   k5. touch();

   Real eps( 0.0 ), maxEps( 0.0 );
   if( Device :: getDevice() == tnlHostDevice )
   {
      // TODO: implement OpenMP support
      for( Index i = 0; i < size; i ++  )
      {
         Real err = ( Real ) ( tau / 3.0 *
                              fabs( 0.2 * _k1[ i ] +
                                   -0.9 * _k3[ i ] +
                                    0.8 * _k4[ i ] +
                                   -0.1 * _k5[ i ] ) );
         eps = Max( eps, err );
      }
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      const Index block_size = 512;
      const Index grid_size = ( size - 1 ) / block_size + 1;

      computeErrorKernel<<< grid_size, block_size >>>( size, tau, _k1, _k3, _k4, _k5, _kAux );
      cudaThreadSynchronize();
      eps = tnlMax( kAux );
#endif
   }
   :: MPIAllreduce( eps, maxEps, 1, MPI_MAX, this -> solver_comm );
   return maxEps;
}

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
void tnlMersonSolver< Problem, Mesh, Real, Device, Index > :: computeNewTimeLevel( Mesh& u,
                                                                                   Real tau,
                                                                                   Real& currentResidue )
{
   Real localResidue = Real( 0.0 );
   Index size = k1. getSize();
   Real* _u = u. getData();
   Real* _k1 = k1. getData();
   Real* _k4 = k4. getData();
   Real* _k5 = k5. getData();

   /****
    * Compute data transfers statistics
    */
   u. touch();
   k1. touch();
   k4. touch();
   k5. touch();

   if( Device :: getDevice() == tnlHostDevice )
   {
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:localResidue) firstprivate( size, _u, _k1, _k4, _k5, tau )
#endif
      for( Index i = 0; i < size; i ++ )
      {
         const Real add = tau / 6.0 * ( _k1[ i ] + 4.0 * _k4[ i ] + _k5[ i ] );
         _u[ i ] += add;
         localResidue += fabs( ( Real ) add );
      }
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      const Index block_size = 512;
      const Index grid_size = ( size - 1 ) / block_size + 1;

      updateU<<< grid_size, block_size >>>( size, tau, _k1, _k4, _k5, _u );
      cudaThreadSynchronize();
      localResidue = 0.0;
#endif
   }
   localResidue /= tau * ( Real ) size;
   :: MPIAllreduce( localResidue, currentResidue, 1, MPI_SUM, this -> solver_comm );

}

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
void tnlMersonSolver< Problem, Mesh, Real, Device, Index > :: writeGrids( const Mesh& u )
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

template< typename Real, typename Index >
__global__ void computeK2Arg( const Index size,
                              const Real tau,
                              const Real* u,
                              const Real* k1,
                              Real* k2_arg )
{
   int i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      k2_arg[ i ] = u[ i ] + tau * ( 1.0 / 3.0 * k1[ i ] );
}

template< typename Real, typename Index >
__global__ void computeK3Arg( const Index size,
                              const Real tau,
                              const Real* u,
                              const Real* k1,
                              const Real* k2,
                              Real* k3_arg )
{
   Index i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      k3_arg[ i ] = u[ i ] + tau * 1.0 / 6.0 * ( k1[ i ] + k2[ i ] );
}

template< typename Real, typename Index >
__global__ void computeK4Arg( const Index size,
                              const Real tau,
                              const Real* u,
                              const Real* k1,
                              const Real* k3,
                              Real* k4_arg )
{
   Index i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      k4_arg[ i ] = u[ i ] + tau * ( 0.125 * k1[ i ] + 0.375 * k3[ i ] );
}

template< typename Real, typename Index >
__global__ void computeK5Arg( const Index size,
                              const Real tau,
                              const Real* u,
                              const Real* k1,
                              const Real* k3,
                              const Real* k4,
                              Real* k5_arg )
{
   Index i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      k5_arg[ i ] = u[ i ] + tau * ( 0.5 * k1[ i ] - 1.5 * k3[ i ] + 2.0 * k4[ i ] );
}

template< typename Real, typename Index >
__global__ void computeErrorKernel( const Index size,
                                    const Real tau,
                                    const Real* k1,
                                    const Real* k3,
                                    const Real* k4,
                                    const Real* k5,
                                    Real* err )
{
   Index i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      err[ i ] = 1.0 / 3.0 *  tau * fabs( 0.2 * k1[ i ] +
                                         -0.9 * k3[ i ] +
                                          0.8 * k4[ i ] +
                                         -0.1 * k5[ i ] );
}

template< typename Real, typename Index >
__global__ void updateU( const Index size,
                         const Real tau,
                         const Real* k1,
                         const Real* k4,
                         const Real* k5,
                         Real* u )
{
        Index i = blockIdx. x * blockDim. x + threadIdx. x;
        if( i < size )
                u[ i ] += 1.0 / 6.0 * tau * ( k1[ i ] + 4.0 * k4[ i ] + k5[ i ] );
}

#endif

#endif
