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
#include <diff/tnlExplicitSolver.h>

#ifdef HAVE_CUDA
void computeK2Arg( const int size, const float* u, const float* k1, float* k2_arg );
void computeK2Arg( const int size, const double* u, const double* k1, double* k2_arg );
void computeK3Arg( const int size, const float* u, const float* k1, const float* k2, float* k3_arg );
void computeK3Arg( const int size, const double* u, const double* k1, const double* k2, double* k3_arg );
void computeK4Arg( const int size, const float* u, const float* k1, const float* k3, float* k4_arg );
void computeK4Arg( const int size, const double* u, const double* k1, const double* k3, double* k4_arg );
void computeK5Arg( const int size, const float* u, const float* k1, const float* k3, const float* k4, float* k5_arg );
void computeK5Arg( const int size, const double* u, const double* k1, const double* k3, const double* k4, double* k5_arg );
#endif

template< class GRID, class SCHEME, typename T = double > class tnlMersonSolver : public tnlExplicitSolver< GRID, SCHEME, T >
{
   public:

   tnlMersonSolver( const GRID& v )
   {
      k1 = new GRID( v );
      k2 = new GRID( v );
      k3 = new GRID( v );
      k4 = new GRID( v );
      k5 = new GRID( v );
      k_tmp = new GRID( v );
      if( ! k1 || ! k2 || ! k3 || ! k4 || ! k5 || ! k_tmp )
      {
         cerr << "Unable to allocate supporting structures for the Merson solver." << endl;
         abort();
      };
      k1 -> Zeros();
      k2 -> Zeros();
      k3 -> Zeros();
      k4 -> Zeros();
      k5 -> Zeros();
      k_tmp -> Zeros();
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
      T* _k1 = k1 -> Data();
      T* _k2 = k2 -> Data();
      T* _k3 = k3 -> Data();
      T* _k4 = k4 -> Data();
      T* _k5 = k5 -> Data();
      T* _k_tmp = k_tmp -> Data();
      T* _u = u. Data();
           
      tnlExplicitSolver< GRID, SCHEME, T > :: iteration = 0;
      double& _time = tnlExplicitSolver< GRID, SCHEME, T > :: time;  
      double& _residue = tnlExplicitSolver< GRID, SCHEME, T > :: residue;  
      int& _iteration = tnlExplicitSolver< GRID, SCHEME, T > :: iteration;
      const double size_inv = 1.0 / ( double ) u. GetSize();
      
      const int block_size = 512;
      const int grid_size = ( size - 1 ) / 512 + 1;

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

         computeK2Arg( size, block_size, grid_size, tau, _u, _k1, _k_tmp );
         scheme. GetExplicitRHSCUDA( _time + tau_3, *k_tmp, *k2 );
         
         computeK3Arg( size, block_size, grid_size, tau, _u, _k1, _k2, _k_tmp );
         scheme. GetExplicitRHSCUDA( _time + tau_3, *k_tmp, *k3 );
         
         computeK4Arg( size, block_size, grid_size, tau, _u, _k1, _k3, _k_tmp );
         scheme. GetExplicitRHSCUDA( _time + 0.5 * _tau, *k_tmp, *k4 );
         
         computeK5Arg( size, block_size, grid_size, tau, _k1, _k3, _k4, _k_tmp );
         scheme. GetExplicitRHSCUDA( _time + _tau, *k_tmp, *k5 );
   
         double eps( 0.0 ), max_eps( 0.0 );
         if( adaptivity )
         {
            for( i = 0; i < size; i ++  )
            {
               eps = Max( eps, _tau / 3.0 * 
                                 fabs( 0.2 * _k1[ i ] +
                                      -0.9 * _k3[ i ] + 
                                       0.8 * _k4[ i ] +
                                      -0.1 * _k5[ i ] ) );
            }
            :: MPIAllreduce( eps, max_eps, 1, MPI_MAX, tnlExplicitSolver< GRID, SCHEME, T > :: solver_comm );
            //if( MPIGetRank() == 0 )
            //   cout << "eps = " << eps << "       " << endl; 
            //  
         }
         //cout << endl << "max_eps = " << max_eps << endl;
         if( ! adaptivity || max_eps < adaptivity )
         {
            double last_residue = _residue;
            double loc_residue = 0.0;

            updateU( size, tau, _k1, _k4, _k5 );
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
      delete k1;
      delete k2;
      delete k3;
      delete k4;
      delete k5;
      delete k_tmp;
   };

   protected:
   
   GRID *k1, *k2, *k3, *k4, *k5, *k_tmp;

   double adaptivity;
};

#endif
