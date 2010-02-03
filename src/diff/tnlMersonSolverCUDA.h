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

#ifndef tnlMersonSolverCUDAH
#define tnlMersonSolverCUDAH

#include <math.h>
#include <core/tnl-cuda-kernels.cu.h>
#include <diff/tnlExplicitSolver.h>

#ifdef HAVE_CUDA
void computeK2Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const float& tau,
                   const float* u,
                   const float* k1,
                   float* k2_arg );
void computeK2Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const double& tau,
                   const double* u,
                   const double* k1,
                   double* k2_arg );
void computeK3Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const float& tau,
                   const float* u,
                   const float* k1,
                   const float* k2,
                   float* k3_arg );
void computeK3Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const double& tau,
                   const double* u,
                   const double* k1,
                   const double* k2,
                   double* k3_arg );
void computeK4Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const float& tau,
                   const float* u,
                   const float* k1,
                   const float* k3,
                   float* k4_arg );
void computeK4Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const double& tau,
                   const double* u,
                   const double* k1,
                   const double* k3,
                   double* k4_arg );
void computeK5Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const float& tau,
                   const float* u,
                   const float* k1,
                   const float* k3,
                   const float* k4,
                   float* k5_arg );
void computeK5Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const double& tau,
                   const double* u,
                   const double* k1,
                   const double* k3,
                   const double* k4,
                   double* k5_arg );
void computeErr( const int size,
                 const int block_size,
                 const int grid_size,
                 const float& tau,
                 const float* _k1,
                 const float* _k3,
                 const float* _k4,
                 const float* _k5,
                 float* _err );
void computeErr( const int size,
                 const int block_size,
                 const int grid_size,
                 const double& tau,
                 const double* _k1,
                 const double* _k3,
                 const double* _k4,
                 const double* _k5,
                 double* _err );
void updateU( const int size,
              const int block_size,
              const int grid_size,
              const float& tau,
              const float* k1,
              const float* k4,
              const float* k5,
              float* u );
void updateU( const int size,
              const int block_size,
              const int grid_size,
              const double& tau,
              const double* k1,
              const double* k4,
              const double* k5,
              double* u );


#endif

template< class GRID, class SCHEME, typename T = double > class tnlMersonSolverCUDA : public tnlExplicitSolver< GRID, SCHEME, T >
{
   public:

   tnlMersonSolverCUDA( const GRID& v )
   : adaptivity( 1.0e-5 )
   {
#ifdef HAVE_CUDA
      k1. SetNewDimensions( v );
      k1. SetNewDomain( v );
      k2. SetNewDimensions( v );
      k2. SetNewDomain( v );
      k3. SetNewDimensions( v );
      k3. SetNewDomain( v );
      k4. SetNewDimensions( v );
      k4. SetNewDomain( v );
      k5. SetNewDimensions( v );
      k5. SetNewDomain( v );
      k_tmp. SetNewDimensions( v );
      k_tmp. SetNewDomain( v );
      err. SetNewDimensions( v );
      err. SetNewDomain( v );
      if( ! k1 || ! k2 || ! k3 || ! k4 || ! k5 || ! k_tmp || ! err )
      {
         cerr << "Unable to allocate supporting structures for the Merson solver." << endl;
         abort();
      };
      tnlExplicitSolver< GRID, SCHEME, T > :: tau = 1.0;
#else
	cerr << "CUDA is not supported on this system." << endl;
#endif
   };

   tnlString GetType() const
   {
      T t;
      GRID grid;
      return tnlString( "tnlMersonSolver< " ) + grid. GetType() + 
             tnlString( ", " ) + GetParameterType( t ) + tnlString( " >" );
   };

   void SetAdaptivity( const T& a )
   {
      adaptivity = a;
   };
   
   bool Solve( SCHEME& scheme,
               GRID& u,
               const T& stop_time,
               const T& max_res,
               const int max_iter = -1 )
   {
#ifdef HAVE_CUDA
      T* _k1 = k1. Data();
      T* _k2 = k2. Data();
      T* _k3 = k3. Data();
      T* _k4 = k4. Data();
      T* _k5 = k5. Data();
      T* _k_tmp = k_tmp. Data();
      T* _u = u. Data();
      T* _err = err. Data();
           
      tnlExplicitSolver< GRID, SCHEME, T > :: iteration = 0;
      T& _time = tnlExplicitSolver< GRID, SCHEME, T > :: time;
      T& _residue = tnlExplicitSolver< GRID, SCHEME, T > :: residue;
      int& _iteration = tnlExplicitSolver< GRID, SCHEME, T > :: iteration;
      const T size_inv = 1.0 / ( T ) u. GetSize();
      
      const int size = u. GetSize();
      const int block_size = 512;
      const int grid_size = ( size - 1 ) / block_size + 1;

      T _tau = tnlExplicitSolver< GRID, SCHEME, T > :: tau;
      if( _time + _tau > stop_time ) _tau = stop_time - _time;
      if( _tau == 0.0 ) return true;

      if( tnlExplicitSolver< GRID, SCHEME, T > :: verbosity > 0 )
         tnlExplicitSolver< GRID, SCHEME, T > :: PrintOut();

      // DEBUG
      /*tnlGrid2D< T > hostAux;
      hostAux. SetNewDimensions( u. GetXSize(), u. GetYSize() );
      hostAux. SetNewDomain( u. GetAx(), u. GetBx(), u. GetAy(), u. GetBy() );*/

      while( 1 )
      {

         int i;
         int size = k1. GetSize();
         assert( size == u. GetSize() );
         
         T tau_3 = _tau / 3.0;

         scheme. GetExplicitRHSCUDA( _time, u, k1 );

         computeK2Arg( size, block_size, grid_size, _tau, _u, _k1, _k_tmp );
         scheme. GetExplicitRHSCUDA( _time + tau_3, k_tmp, k2 );
         
         computeK3Arg( size, block_size, grid_size, _tau, _u, _k1, _k2, _k_tmp );
         scheme. GetExplicitRHSCUDA( _time + tau_3, k_tmp, k3 );
         
         computeK4Arg( size, block_size, grid_size, _tau, _u, _k1, _k3, _k_tmp );
         scheme. GetExplicitRHSCUDA( _time + 0.5 * _tau, k_tmp, k4 );
         
         computeK5Arg( size, block_size, grid_size, _tau, _u, _k1, _k3, _k4, _k_tmp );
         scheme. GetExplicitRHSCUDA( _time + _tau, k_tmp, k5 );
   
         T eps( 0.0 ), max_eps( 0.0 );
         if( adaptivity )
         {
            computeErr( size, block_size, grid_size, _tau, _k1, _k3, _k4, _k5, _err );
            tnlCUDAReductionMax( size, _err, max_eps ); // TODO: allocate auxiliary array for the reduction

            /*hostAux. copyFrom( err );
            T seq_max_eps = 0.0;
            for( int i = 0; i < hostAux. GetSize(); i ++ )
            	seq_max_eps = :: Max( seq_max_eps, hostAux[ i ] );
            if( max_eps != seq_max_eps )
            	abort();*/

         }

         //cout << endl << "max_eps = " << max_eps << endl;
         if( ! adaptivity || max_eps < adaptivity )
         {
            T last_residue = _residue;
            T loc_residue = 0.0;

            updateU( size,  block_size, grid_size, _tau, _k1, _k4, _k5, _u );
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
         if( _iteration == max_iter ) return false;
      }
#else
      cerr << "CUDA is not supported on this system." << endl;
      return false;
#endif
   };

   ~tnlMersonSolverCUDA()
   {
   };

   protected:
   
   GRID k1, k2, k3, k4, k5, k_tmp, err;

   T adaptivity;
};

#endif
