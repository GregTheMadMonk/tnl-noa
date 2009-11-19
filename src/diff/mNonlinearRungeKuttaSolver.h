/***************************************************************************
                          mNonlinearRungeKuttaSolver.h  -  description
                             -------------------
    begin                : 2007/07/06
    copyright            : (C) 2007 by Tomá¹ Oberhuber
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

#ifndef mNonlinearRungeKuttaSolverH
#define mNonlinearRungeKuttaSolverH

#include <math.h>
#include <diff/mExplicitSolver.h>

template< class GRID, class SCHEME, typename T = double > class mNonlinearRungeKuttaSolver : public mExplicitSolver< GRID, SCHEME, T >
{
   public:

   mNonlinearRungeKuttaSolver( const GRID& v )
   {
      k1 = new GRID( v );
      k2 = new GRID( v );
      k3 = new GRID( v );
      k4 = new GRID( v );
      k5 = new GRID( v );
      k_tmp = new GRID( v );
      if( ! k1 || ! k2 || ! k3 || ! k4 || ! k5 || ! k_tmp )
      {
         cerr << "Unable to allocate supporting structures for Merson solver." << endl;
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
      return tnlString( "mNonLinearRungeKuttaSolver< " ) + grid. GetType() + 
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
         const long int size = k1 -> GetSize();
         assert( size == u. GetSize() );
         
         const T tau_3 = _tau / 3.0;

         scheme. GetExplicitRHS( _time, u, *k1 );

         for( i = 0; i < size; i ++ )
            _k_tmp[ i ] = _u[ i ] + _tau * ( 2.0 / 9.0 * _k1[ i ] ); 
         scheme. GetExplicitRHS( _time + 2.0 / 9.0 * _tau, *k_tmp, *k2 );      
         
         for( i = 0; i < size; i ++ )
            _k_tmp[ i ] = _u[ i ] + _tau * ( 1.0 / 12.0 * _k1[ i ] + 1.0 / 4.0 * _k2[ i ] ); 
         scheme. GetExplicitRHS( _time + 1.0 / 3.0 * _tau, *k_tmp, *k3 );      
         
         for( i = 0; i < size; i ++ )
            _k_tmp[ i ] = _u[ i ] + _tau * 3.0 / 128.0 * ( 23.0 * _k1[ i ] - 81.0 * _k2[ i ] + 90.0 * _k3[ i ] ); 
         scheme. GetExplicitRHS( _time + 0.75 * _tau, *k_tmp, *k4 );      
         
         for( i = 0; i < size; i ++ )
            _k_tmp[ i ] = _u[ i ] + _tau * 9.0 / 10000.0 * ( -345.0 * _k1[ i ] + 2025.0 * _k2[ i ] - 1224.0 * _k3[ i ] + 544.0 * _k4[ i ] ); 
         scheme. GetExplicitRHS( _time + 9.0 / 10.0 * _tau, *k_tmp, *k5 );      
   
         double eps( 0.0 ), max_eps( 0.0 );
         if( adaptivity )
         {
            for( i = 0; i < size; i ++  )
            {
               const T q = -1.0 / 18.0 * _k1[ i ] +
                            27.0 / 170.0 * _k3[ i ] +
                           -4.0 / 15.0 * _k4[ i ] +
                            25.0 / 153.0 * _k5[ i ];
               const T r =  19.0 / 24.0 * _k1[ i ] +
                           -27.0 / 8.0 * _k2[ i ] +
                            57.0 / 20.0 * _k3[ i ] +
                           -4.0 / 15.0 * _k4[ i ];
               const T s = _k4[ i ] - _k1[ i ];
               if( s != 0 )
                  eps = Max( eps, fabs( _tau * q * r / s  ) );
            }
            :: MPIAllreduce( eps, max_eps, 1, MPI_MAX, mExplicitSolver< GRID, SCHEME, T > :: solver_comm );
            //if( MPIGetRank() == 0 )
            //   cout << "eps = " << eps << "       " << endl; 
         }
         if( ! adaptivity || max_eps < adaptivity )
         {
            double last_residue = _residue;
            _residue = 0.0;
            for( i = 0; i < size; i ++ )
            {
               // this does not have to be in double precision
               const T add = _tau * ( 17.0 / 162.0 * _k1[ i ] +
                                      81.0 / 170.0 * _k3[ i ] +
                                      32.0 / 135.0 * _k4[ i ] +
                                      250.0 / 1377.0 * _k5[ i ] );
               _u[ i ] += add; 
               _residue += fabs( ( double ) add );
            }
            if( _tau + _time == stop_time ) _residue = last_residue;  // fixing strange values of res. at the last iteration
            else
             {
                double loc_residue = _residue / _tau * size_inv;
                :: MPIAllreduce( loc_residue, _residue, 1, MPI_SUM, mExplicitSolver< GRID, SCHEME, T > :: solver_comm );
             }
            _time += _tau;
            _iteration ++;
         }
         if( adaptivity && max_eps != 0.0 ) _tau *= 0.8 * pow( adaptivity / max_eps, 0.2 );
         :: MPIBcast( _tau, 1, 0, mExplicitSolver< GRID, SCHEME, T > :: solver_comm );

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

   ~mNonlinearRungeKuttaSolver()
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
