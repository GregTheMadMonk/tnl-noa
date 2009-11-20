/***************************************************************************
                          mFehlbergSolver.h  -  description
                             -------------------
    begin                : 2007/06/19
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

#ifndef mFehlbergSolverH
#define mFehlbergSolverH

#include <math.h>
#include <diff/tnlExplicitSolver.h>

template< class GRID, class SCHEME, typename T = double > class mFehlbergSolver : public tnlExplicitSolver< GRID, SCHEME, T >
{
   public:

   mFehlbergSolver( const GRID& v )
   {
      k1 = new GRID( v );
      k2 = new GRID( v );
      k3 = new GRID( v );
      k4 = new GRID( v );
      k5 = new GRID( v );
      k6 = new GRID( v );
      k_tmp = new GRID( v );
      if( ! k1 || ! k2 || ! k3 || ! k4 || ! k5 || ! k6 || ! k_tmp )
      {
         cerr << "Unable to allocate supporting structures for Merson solver." << endl;
         abort();
      };
   };

   tnlString GetType() const
   {
      T t;
      GRID grid;
      return tnlString( "mFehlbergSolver< " ) + grid. GetType() + 
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
      T* _k6 = k6 -> Data();
      T* _k_tmp = k_tmp -> Data();
      T* _u = u. Data();
           
      tnlExplicitSolver< GRID, SCHEME, T > :: iteration = 0;
      double& _time = tnlExplicitSolver< GRID, SCHEME, T > :: time;  
      double& _residue = tnlExplicitSolver< GRID, SCHEME, T > :: residue;  
      long int& _iteration = tnlExplicitSolver< GRID, SCHEME, T > :: iteration;
      
      T _tau = tnlExplicitSolver< GRID, SCHEME, T > :: tau;
      if( _time + _tau > stop_time )
         _tau = stop_time - _time;
      if( _tau == 0.0 ) return true;

      while( 1 )
      {
         long int i;
         const long int size = k1 -> Size();
         assert( size == u. Size() );
         
         scheme. GetExplicitRHS( _time, u, *k1 );

         for( i = 0; i < size; i ++ )
         {
            _k1[ i ] *= _tau;
            _k_tmp[ i ] = _u[ i ] + 0.25 * _k1[ i ]; 
         }
         scheme. GetExplicitRHS( _time + 0.25 * _tau, *k_tmp, *k2 );      
         
         for( i = 0; i < size; i ++ )
         {
            _k2[ i ] *= _tau;
            _k_tmp[ i ] = _u[ i ] + 3.0 / 32.0 * _k1[ i ] + 
                                    9.0 / 32.0 * _k2[ i ]; 
         }
         scheme. GetExplicitRHS( _time + 3.0 / 8.0 * _tau, *k_tmp, *k3 );      
         
         for( i = 0; i < size; i ++ )
         {
            _k3[ i ] *= _tau;
            _k_tmp[ i ] = _u[ i ] + 1932.0 / 2197.0 * _k1[ i ] - 
                                    7200.0 / 2197.0 * _k2[ i ] +
                                    7296.0 / 2197.0 * _k3[ i ]; 
         }      
         scheme. GetExplicitRHS( _time + 12.0 / 13.0 * _tau, *k_tmp, *k4 );      
         
         for( i = 0; i < size; i ++ )
         {
            _k4[ i ] *= _tau;
            _k_tmp[ i ] = _u[ i ] + 439.0 / 216.0 * _k1[ i ] -
                                    8.0 * _k2[ i ] +
                                    3680.0 / 513.0 * _k3[ i ] - 
                                    845.0 / 4104.0 * _k4[ i ]; 
         }
         scheme. GetExplicitRHS( _time + _tau, *k_tmp, *k5 );      
         
         for( i = 0; i < size; i ++ )
         {
            _k5[ i ] *= _tau;
            _k_tmp[ i ] = _u[ i ] - 8.0 / 27.0 * _k1[ i ] +
                                    2.0 * _k2[ i ] -
                                    3544.0 / 2565.0 * _k3[ i ] +
                                    1859.0 / 4104.0 * _k4[ i ] -
                                    11.0 / 40.0 * _k5[ i ]; 
         }
         scheme. GetExplicitRHS( _time + 0.5 * _tau, *k_tmp, *k6 );      
         double err( 0.0 );
         
         for( i = 0; i < size; i ++ )
         {
            _k6[ i ] *= _tau;
            _k_tmp[ i ] =  16.0 / 135.0 * _k1[ i ] +
                           6656.0 / 12825.0 * _k3[ i ] +
                           28561.0 / 56430.0 * _k4[ i ] -
                           9.0 / 50.0 * _k5[ i ] +
                           2.0 / 55.0 * _k6[ i ];
            err = Max( err, fabs(  _k_tmp[ i ] -
                                   25.0 / 216.0 * _k1[ i ] -
                                   1408.0 / 2565.0 * _k3[ i ] -
                                   2197.0 / 4104.0 * _k4[ i ] -
                                   0.2 * _k5[ i ] ) );
         }
   
         if( err < adaptivity )
         {
            for( i = 0; i < size; i ++ )
            {
               u[ i ] += _k_tmp[ i ];
               _residue += fabs( _k_tmp[ i ] );
            }
         }
         _residue /= _tau * ( double ) ( size );
         _time += _tau;
         _tau *= pow( 0.5 * adaptivity * _tau / err, 0.25 );
         if( _time + _tau > stop_time )
            _tau = stop_time - _time; //we don't want to keep such tau
         else tnlExplicitSolver< GRID, SCHEME, T > :: tau = _tau;
         _iteration ++;
         
         tnlExplicitSolver< GRID, SCHEME, T > :: PrintOut();
         
         if( _time == stop_time ) return true;
         if( max_res && _residue < max_res ) return true;
         //if( max_iter && _iteration == max_iter ) return false;
      }
   };

   ~mFehlbergSolver()
   {
      delete k1;
      delete k2;
      delete k3;
      delete k4;
      delete k5;
      delete k6;
      delete k_tmp;
   };

   protected:
   
   GRID *k_tmp, *k1, *k2, *k3, *k4, *k5, *k6;

   double adaptivity;
};
#endif
