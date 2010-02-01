/***************************************************************************
                          tnlMersonSolverCUDA.h  -  description
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
#include <core/tnl-cuda-kernels.h>
#include <diff/tnlExplicitSolver.h>

#ifdef HAVE_CUDA
// TODO: remove this
/*void computeK2Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const float* u,
                   const float* k1,
                   float* k2_arg );
void computeK2Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const double* u,
                   const double* k1,
                   double* k2_arg );
void computeK3Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const float* u,
                   const float* k1,
                   const float* k2,
                   float* k3_arg );
void computeK3Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const double* u,
                   const double* k1,
                   const double* k2,
                   double* k3_arg );
void computeK4Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const float* u,
                   const float* k1,
                   const float* k3,
                   float* k4_arg );
void computeK4Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const double* u,
                   const double* k1,
                   const double* k3,
                   double* k4_arg );
void computeK5Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const float* u,
                   const float* k1,
                   const float* k3,
                   const float* k4,
                   float* k5_arg );
void computeK5Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const double* u,
                   const double* k1,
                   const double* k3,
                   const double* k4,
                   double* k5_arg );
void updateU( const int size,
              const int block_size,
              const int grid_size,
              const float tau,
              const float* k1,
              const float* k4,
              const float* k5,
              float* u );
void updateU( const int size,
              const int block_size,
              const int grid_size,
              const double tau,
              const double* k1,
              const double* k4,
              const double* k5,
              double* u );*/

template< class T > __global__ void computeK2ArgKernel( const int size, const T tau, const T* u, const T* k1, T* k2_arg )
{
   int i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      k2_arg[ i ] = u[ i ] + tau * ( 1.0 / 3.0 * k1[ i ] );
}

template< class T > __global__ void computeK3ArgKernel( const int size, const T tau, const T* u, const T* k1, const T* k2, T* k3_arg )
{
   int i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      k3_arg[ i ] = u[ i ] + tau * 1.0 / 6.0 * ( k1[ i ] + k2[ i ] );
}

template< class T > __global__ void computeK4ArgKernel( const int size, const T tau, const T* u, const T* k1, const T* k3, T* k4_arg )
{
   int i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      k4_arg[ i ] = u[ i ] + tau * ( 0.125 * k1[ i ] + 0.375 * k3[ i ] );
}

template< class T > __global__ void computeK5ArgKernel( const int size, const T tau, const T* u, const T* k1, const T* k3, const T* k4, T* k5_arg )
{
   int i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      k5_arg[ i ] = u[ i ] + tau * ( 0.5 * k1[ i ] - 1.5 * k3[ i ] + 2.0 * k4[ i ] );
}

template< class T > __global__ void computeErrKernel( const int size, const T tau, const T* _k1, const T* _k3, const T* _k4, const T* _k5, T* err )
{
   int i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      err[ i ] = 1.0 / 3.0 *  tau * fabs( 0.2 * k1[ i ] +
                                         -0.9 * k3[ i ] +
                                          0.8 * k4[ i ] +
                                         -0.1 * k5[ i ] ) );
}

template< class T > __global__ void updateUKernel( const int size, const T tau, const T* k1, const T* k4, const T* k5, T* u )
{
        int i = blockIdx. x * blockDim. x + threadIdx. x;
        if( i < size )
                u[ i ] += tau / 6.0 * ( k1[ i ] + 4.0 * k4[ i ] + k5[ i ] );
}

#endif

template< class GRID, class SCHEME, typename T = double > class tnlMersonSolverCUDA : public tnlExplicitSolver< GRID, SCHEME, T >
{
   public:

   tnlMersonSolver( const GRID& v )
   {
      k1. SetNewDimension( v );
      k1. SetNewDomain( v );
      k2. SetNewDimension( v );
      k2. SetNewDomain( v );
      k3. SetNewDimension( v );
      k3. SetNewDomain( v );
      k4. SetNewDimension( v );
      k4. SetNewDomain( v );
      k5. SetNewDimension( v );
      k5. SetNewDomain( v );
      k_tmp. SetNewDimension( v );
      k_tmp. SetNewDomain( v );
      err. SetNewDimension( v );
      err. SetNewDomain( v );
      if( ! k1 || ! k2 || ! k3 || ! k4 || ! k5 || ! k_tmp || ! err )
      {
         cerr << "Unable to allocate supporting structures for the Merson solver." << endl;
         abort();
      };
      k1. Zeros();
      k2. Zeros();
      k3. Zeros();
      k4. Zeros();
      k5. Zeros();
      k_tmp. Zeros();
      err. Zeros();
   };

   tnlString GetType() const
   {
      T t;
      GRID grid;
      return tnlString( "tnlMersonSolver< " ) + grid. GetType() + 
             tnlString( ", " ) + GetParameterType( t ) + tnlString( " >" );
   };

   void SetAdaptivity( const double& a )
   {
      adaptivity = a;
   };
   
   bool Solve( SCHEME& scheme,
               GRID& u,
               const double& stop_time,
               const double& max_res,
               const int max_iter )
   {
      T* _k1 = k1. Data();
      T* _k2 = k2. Data();
      T* _k3 = k3. Data();
      T* _k4 = k4. Data();
      T* _k5 = k5. Data();
      T* _k_tmp = k_tmp. Data();
      T* _u = u. Data();
      T* _err = err. Data();
           
      tnlExplicitSolver< GRID, SCHEME, T > :: iteration = 0;
      double& _time = tnlExplicitSolver< GRID, SCHEME, T > :: time;  
      double& _residue = tnlExplicitSolver< GRID, SCHEME, T > :: residue;  
      int& _iteration = tnlExplicitSolver< GRID, SCHEME, T > :: iteration;
      const double size_inv = 1.0 / ( double ) u. GetSize();
      
      const int block_size = 512;
      const int grid_size = ( size - 1 ) / block_size + 1;

      T _tau = tnlExplicitSolver< GRID, SCHEME, T > :: tau;
      if( _time + _tau > stop_time ) _tau = stop_time - _time;
      if( _tau == 0.0 ) return true;

      if( tnlExplicitSolver< GRID, SCHEME, T > :: verbosity > 0 )
         tnlExplicitSolver< GRID, SCHEME, T > :: PrintOut();
      while( 1 )
      {

         int i;
         int size = k1 -> GetSize();
         assert( size == u. GetSize() );
         
         T tau_3 = _tau / 3.0;

         scheme. GetExplicitRHSCUDA( _time, u, *k1 );

         computeK2ArgKernel<<< block_size, grid_size >>>( size, tau, _u, _k1, _k_tmp );
         scheme. GetExplicitRHSCUDA( _time + tau_3, *k_tmp, *k2 );
         
         computeK3ArgKernel<<< block_size, grid_size >>>( size, tau, _u, _k1, _k2, _k_tmp );
         scheme. GetExplicitRHSCUDA( _time + tau_3, *k_tmp, *k3 );
         
         computeK4ArgKernel<<< block_size, grid_size >>>( size, tau, _u, _k1, _k3, _k_tmp );
         scheme. GetExplicitRHSCUDA( _time + 0.5 * _tau, *k_tmp, *k4 );
         
         computeK5Arg<<< block_size, grid_size >>>( size, tau, _k1, _k3, _k4, _k_tmp );
         scheme. GetExplicitRHSCUDA( _time + _tau, *k_tmp, *k5 );
   
         double eps( 0.0 ), max_eps( 0.0 );
         if( adaptivity )
         {
            computeErrKernel<<< block_size, grid_size >>>( size, tau, _k1, _k3, _k4, _k5, _err );
            tnlCUDAReduction( size, err, max_eps ); // TODO: allocate auxiliary array for the reduction
         }
         //cout << endl << "max_eps = " << max_eps << endl;
         if( ! adaptivity || max_eps < adaptivity )
         {
            double last_residue = _residue;
            double loc_residue = 0.0;

            updateU( size, block_size, grid_size, tau, _k1, _k4, _k5 );
            // TODO: implement loc_residue - if possible

            if( _tau + _time == stop_time ) _residue = last_residue;  // fixing strange values of res. at the last iteration
            else
            {
                loc_residue /= _tau * size_inv;
                :: MPIAllreduce( loc_residue, _residue, 1, MPI_SUM, tnlExplicitSolver< GRID, SCHEME, T > :: solver_comm );
            }
            _time += _tau;
            _iteration ++;
         }
         if( adaptivity && max_eps != 0.0 )
         {
            _tau *= 0.8 * pow( adaptivity / max_eps, 0.2 );
            :: MPIBcast( _tau, 1, 0, tnlExplicitSolver< GRID, SCHEME, T > :: solver_comm );
         }

         if( _time + _tau > stop_time )
            _tau = stop_time - _time; //we don't want to keep such tau
         else tnlExplicitSolver< GRID, SCHEME, T > :: tau = _tau;
         
         if( tnlExplicitSolver< GRID, SCHEME, T > :: verbosity > 1 )
            tnlExplicitSolver< GRID, SCHEME, T > :: PrintOut();
         
         if( _time == stop_time || 
             ( max_res && _residue < max_res ) )
          {
            if( tnlExplicitSolver< GRID, SCHEME, T > :: verbosity > 0 )
               tnlExplicitSolver< GRID, SCHEME, T > :: PrintOut();
             return true;
          }
         //if( max_iter && _iteration == max_iter ) return false;
      }
   };

   ~tnlMersonSolver()
   {
   };

   protected:
   
   GRID k1, k2, k3, k4, k5, k_tmp, err;

   double adaptivity;
};

#endif
