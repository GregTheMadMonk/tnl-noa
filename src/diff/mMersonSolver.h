/***************************************************************************
                          mMersonSolver.h  -  description
                             -------------------
    begin                : 2007/06/16
    copyright            : (C) 2007 by Tomá¹ Oberhuber
    email                : oberhuber@seznam.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef mMersonSolverH
#define mMersonSolverH

#include <math.h>
#include <mcore.h>
#include "mExplicitSolver.h"

template< class GRID, class SCHEME, typename T = double > class mMersonSolver : public mExplicitSolver< GRID, SCHEME, T >
{
   public:

   mMersonSolver( const GRID& v )
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

   mString GetType() const
   {
      T t;
      GRID grid;
      return mString( "mMersonSolver< " ) + grid. GetType() + 
             mString( ", " ) + GetParameterType( t ) + mString( " >" );
   };

   void SetAdaptivity( const double& a )
   {
      adaptivity = a;
   };
   
   bool Solve( SCHEME& scheme,
               GRID& u,
               const double& stop_time,
               const double& max_res,
               const long int max_iter )
   {
      T* _k1 = k1 -> Data();
      T* _k2 = k2 -> Data();
      T* _k3 = k3 -> Data();
      T* _k4 = k4 -> Data();
      T* _k5 = k5 -> Data();
      T* _k_tmp = k_tmp -> Data();
      T* _u = u. Data();
           
      mExplicitSolver< GRID, SCHEME, T > :: iteration = 0;
      double& _time = mExplicitSolver< GRID, SCHEME, T > :: time;  
      double& _residue = mExplicitSolver< GRID, SCHEME, T > :: residue;  
      long int& _iteration = mExplicitSolver< GRID, SCHEME, T > :: iteration;
      const double size_inv = 1.0 / ( double ) u. GetSize();
      
      T _tau = mExplicitSolver< GRID, SCHEME, T > :: tau;
      if( _time + _tau > stop_time ) _tau = stop_time - _time;
      if( _tau == 0.0 ) return true;

      if( mExplicitSolver< GRID, SCHEME, T > :: verbosity > 0 )
         mExplicitSolver< GRID, SCHEME, T > :: PrintOut();
      while( 1 )
      {

         long int i;
         long int size = k1 -> GetSize();
         assert( size == u. GetSize() );
         
         T tau_3 = _tau / 3.0;

         scheme. GetExplicitRHS( _time, u, *k1 );

#ifdef HAVE_OPENMP
#pragma omp parallel for private( i ) firstprivate( size, _k_tmp, _u, _k1, _tau, tau_3 ) 
#endif
         for( i = 0; i < size; i ++ )
            _k_tmp[ i ] = _u[ i ] + _tau * ( 1.0 / 3.0 * _k1[ i ] ); 
         scheme. GetExplicitRHS( _time + tau_3, *k_tmp, *k2 );      
         
#ifdef HAVE_OPENMP
#pragma omp parallel for private( i ) firstprivate( size, _k_tmp, _u, _k1, _k2, _tau, tau_3 ) 
#endif
         for( i = 0; i < size; i ++ )
            _k_tmp[ i ] = _u[ i ] + _tau * 1.0 / 6.0 * ( _k1[ i ] + _k2[ i ] ); 
         scheme. GetExplicitRHS( _time + tau_3, *k_tmp, *k3 );      
         
#ifdef HAVE_OPENMP
#pragma omp parallel for private( i ) firstprivate( size, _k_tmp, _u, _k1, _k3, _tau, tau_3 )
#endif
         for( i = 0; i < size; i ++ )
            _k_tmp[ i ] = _u[ i ] + _tau * ( 0.125 * _k1[ i ] + 0.375 * _k3[ i ] ); 
         scheme. GetExplicitRHS( _time + 0.5 * _tau, *k_tmp, *k4 );      
         
#ifdef HAVE_OPENMP
#pragma omp parallel for private( i ) firstprivate( size, _k_tmp, _u, _k1, _k3, _k4, _tau, tau_3 )
#endif
         for( i = 0; i < size; i ++ )
            _k_tmp[ i ] = _u[ i ] + _tau * ( 0.5 * _k1[ i ] - 1.5 * _k3[ i ] + 2.0 * _k4[ i ] ); 
         scheme. GetExplicitRHS( _time + _tau, *k_tmp, *k5 );      
   
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
            :: MPIAllreduce( eps, max_eps, 1, MPI_MAX, mExplicitSolver< GRID, SCHEME, T > :: solver_comm );
            //if( MPIGetRank() == 0 )
            //   cout << "eps = " << eps << "       " << endl; 
            //  
         }
         //cout << endl << "max_eps = " << max_eps << endl;
         if( ! adaptivity || max_eps < adaptivity )
         {
            double last_residue = _residue;
            double loc_residue = 0.0;
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:loc_residue) firstprivate( size, _k_tmp, _u, _k1,_tau, tau_3 )
#endif
            for( i = 0; i < size; i ++ )
            {
               // this does not have to be in double precision
               //_k_tmp[ i ] = ( 0.5 * ( _k1[ i ] + _k5[ i ] ) + 
               //                2.0 * _k4[ i ] ) * tau_3;
               const T add = _tau / 6.0 * ( _k1[ i ] + 4.0 * _k4[ i ] + _k5[ i ] );
               _u[ i ] += add; 
               loc_residue += fabs( ( double ) add );
            }
            if( _tau + _time == stop_time ) _residue = last_residue;  // fixing strange values of res. at the last iteration
            else
            {
                loc_residue /= _tau * size_inv;
                :: MPIAllreduce( loc_residue, _residue, 1, MPI_SUM, mExplicitSolver< GRID, SCHEME, T > :: solver_comm );
            }
            _time += _tau;
            _iteration ++;
         }
         if( adaptivity && max_eps != 0.0 )
         {
            _tau *= 0.8 * pow( adaptivity / max_eps, 0.2 );
            :: MPIBcast( _tau, 1, 0, mExplicitSolver< GRID, SCHEME, T > :: solver_comm );
         }

         if( _time + _tau > stop_time )
            _tau = stop_time - _time; //we don't want to keep such tau
         else mExplicitSolver< GRID, SCHEME, T > :: tau = _tau;
         
         if( mExplicitSolver< GRID, SCHEME, T > :: verbosity > 1 )
            mExplicitSolver< GRID, SCHEME, T > :: PrintOut();
         
         if( _time == stop_time || 
             ( max_res && _residue < max_res ) )
          {
            if( mExplicitSolver< GRID, SCHEME, T > :: verbosity > 0 )
               mExplicitSolver< GRID, SCHEME, T > :: PrintOut();
             return true;
          }
         //if( max_iter && _iteration == max_iter ) return false;
      }
   };

   ~mMersonSolver()
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
