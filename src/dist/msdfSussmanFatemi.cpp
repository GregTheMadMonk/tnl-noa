/***************************************************************************
                          msdfSussmanFatemi.cpp  -  description
                             -------------------
    begin                : 2008/03/13
    copyright            : (C) 2008 by Tomá¹ Oberhuber
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

#include "msdfSussmanFatemi.h"

#include <config.h>
#include <float.h>
#include <math.h>
#include <core/mfuncs.h>
#include <diff/mEulerSolver.h>
#include <diff/mMersonSolver.h>
#include <diff/mNonlinearRungeKuttaSolver.h>

//--------------------------------------------------------------------------
msdfSussmanFatemi :: msdfSussmanFatemi()
  : _u0( 0 ), 
    _heaviside_prime( 0 ),
    _l( 0 ),
    _mod_grad_u0( 0 ),
    _sign_u0( 0 ),
    _g2( 0 )
{
}
//--------------------------------------------------------------------------
bool msdfSussmanFatemi :: Init( const mParameterContainer& parameters,
                                mGrid2D< double >* _u )
{
   if( ! InitBase( parameters ) ) return false;
   if( ! SetSolver( _u ) ) return false;
   SetInitialCondition( * _u );
   if( ! initial_tau )
   {
      const double h = Min( _u -> GetHx(), _u -> GetHy() );
      initial_tau = 0.5 * h;
   }
   return true;
}
//--------------------------------------------------------------------------
void msdfSussmanFatemi :: SetTimeDiscretisation( const char* str )
{
   if( str )
      time_discretisation. SetString( str );
}
//--------------------------------------------------------------------------
bool msdfSussmanFatemi :: SetSolver( const mGrid2D< double >* u, const char* str ) 
{
   if( str )
      solver_name. SetString( str );
   if( time_discretisation == "explicit" )
   {
      if( explicit_solver ) delete explicit_solver;
      if( solver_name == "euler" )
        explicit_solver = new mEulerSolver< mGrid2D< double >, msdfIterBase, double >( *u );
      if( solver_name == "merson" )
        explicit_solver = new mMersonSolver< mGrid2D< double >, msdfIterBase, double >( *u );
      if( solver_name == "nonlinear-rk" )
        explicit_solver = new mNonlinearRungeKuttaSolver< mGrid2D< double >, msdfIterBase, double >( *u );
      if( ! explicit_solver )
      {
         cerr << "Unable to allocate the solver. " << endl;
         return false;
      }
      if( solver_name == "merson" )
         ( ( mMersonSolver< mGrid2D< double >, msdfIterBase, double >* ) explicit_solver ) -> 
               SetAdaptivity( merson_adaptivity );
      if( solver_name == "nonlinear-rk" )
         ( ( mNonlinearRungeKuttaSolver< mGrid2D< double >, msdfIterBase, double >* ) explicit_solver ) -> 
               SetAdaptivity( merson_adaptivity );
      explicit_solver -> SetVerbosity( verbose );
   }
   return true;
}
//--------------------------------------------------------------------------
void msdfSussmanFatemi :: SetInitialCondition( mGrid2D< double >& u_ini )
{
   if( _u0 ) delete _u0;
   if( _heaviside_prime ) delete _heaviside_prime;
   if( _l ) delete _l;
   if( _mod_grad_u0 ) delete _mod_grad_u0;
   if( _sign_u0 ) delete _sign_u0;
   if( _g2 ) delete _g2;
   global_u = &u_ini;
   u = &u_ini;

   // Store the copy of the initial condition to u0
   _u0 = new mGrid2D< double >( *u );
   const long int x_size = u -> GetXSize();
   const long int y_size = u -> GetYSize();
   long int i, j;
   for( i = 0; i < x_size * y_size; i ++ )
         ( *_u0 )[ i ] = ( *u )[ i ];

   _heaviside_prime = new mGrid2D< double >( *u );
   _l = new mGrid2D< double >( *u );
   _mod_grad_u0 = new mGrid2D< double >( *u );
   _sign_u0 = new mGrid2D< double >( *u );
   _g2 = new mGrid2D< double >( *u );
   double* heaviside_prime = _heaviside_prime -> Data();
   
   // Compute mod grad of u0
   const double* u0 = _u0 -> Data();
   double* mod_grad_u0 = _mod_grad_u0 -> Data();
   const double& hx = _u0 -> GetHx();
   const double& hy = _u0 -> GetHy();
   const double _hx = 1.0 / hx;
   const double _hy = 1.0 / hy;
   const double delta_x = Max( hx, hy );
#ifdef HAVE_OPENMP
#pragma omp parallel for private( i, j ), firstprivate( _hx, _hy, u0, heaviside_prime, mod_grad_u0 )
#endif
   for( i = 0; i < x_size; i ++ )
      for( j = 0; j < y_size; j ++ )
      {
         const long int ij = i * y_size + j;
         const double& u0_ij = u0[ ij ];

         double Dxf( 0.0 ), Dxb( 0.0 ), Dyf( 0.0 ), Dyb( 0.0 );
         if( i > 0 ) Dxb = ( u0_ij - u0[ ij - y_size ] ) * _hx;
         if( i < x_size - 1 ) Dxf = ( u0[ ij + y_size ] - u0_ij ) * _hx;
         if( i == 0 ) Dxb = Dxf;
         if( i == x_size - 1 ) Dxf = Dxb;
         
         if( j > 0 ) Dyb = ( u0_ij - u0[ ij - 1 ] ) * _hy;
         if( j < y_size - 1 ) Dyf = ( u0[ ij + 1 ] - u0_ij ) * _hy;
         if( j == 0 ) Dyb = Dyf;
         if( j == y_size - 1 ) Dyf = Dyb;
         
         double heaviside( 0.0 );
         if( u0_ij > delta_x ) heaviside = 1.0;
         else
            if( u0_ij > - delta_x ) heaviside = 0.5 * ( 1.0 + u0_ij / delta_x + 1.0 / M_PI * sin ( M_PI * u0_ij / delta_x ) );
         double sign_u0 = ( *_sign_u0 )( i, j ) = 2.0 * heaviside - 1.0;

         double Dx( 0.0 ), Dy( 0.0 );
         if( Dxf * sign_u0 < 0.0 && ( Dxf + Dxb ) * sign_u0 < 0.0 ) Dx = Dxf;           
         if( Dxb * sign_u0 > 0.0 && ( Dxf + Dxb ) * sign_u0 > 0.0 ) Dx = Dxb;           
         
         if( Dyf * sign_u0 < 0.0 && ( Dyf + Dyb ) * sign_u0 < 0.0 ) Dy = Dyf;           
         if( Dyb * sign_u0 > 0.0 && ( Dyf + Dyb ) * sign_u0 > 0.0 ) Dy = Dyb;           

         mod_grad_u0[ ij ] = sqrt( Dx * Dx + Dy * Dy );
      
         heaviside_prime[ ij ] = 0.0;
         if( u0_ij > - delta_x && u0_ij < delta_x )
            heaviside_prime[ ij ] = 0.5 / delta_x + 0.5 / delta_x * cos( M_PI * u0_ij / delta_x );
      }
   
   const double int_coeffs[ 3 ][ 3 ] = { { 1.0, 1.0, 1.0 },
                                         { 1.0, 16.0, 1.0 },
                                         { 1.0, 1.0, 1.0 } };
#ifdef HAVE_OPENMP
#pragma omp parallel for private( i, j ), firstprivate( heaviside_prime, int_coeffs, mod_grad_u0 )
#endif
   for( i = 0; i < x_size; i ++ )
      for( j = 0; j < y_size; j ++ )
      {
         const long int ij = i * y_size + j;
         double lambda_ij( 0.0 );
         if( i > 0 && i < x_size - 1 && j > 0 && j < y_size - 1 )
         {
            double g1( 0.0 ), g2( 0.0 );
            long int m, n;
            for( m = -1; m <= 1; m ++ )
               for( n = -1; n <= 1; n ++ )
               {
                  double hv = ( * _heaviside_prime )( i + m, j + n );
                  //g1 += int_coeffs[ m + 1 ][ n + 1 ] * hv * ( * _L )( i + m, j + n );
                  g2 += int_coeffs[ m + 1 ][ n + 1 ] * hv * hv * ( * _mod_grad_u0 )( i + m, j + n );
               }
            ( *_g2 )( i, j ) = g2;
         }
      }
}
//--------------------------------------------------------------------------
void msdfSussmanFatemi :: SetOutputPeriod( const double& t )
{
   output_period = t;
}
//--------------------------------------------------------------------------
void msdfSussmanFatemi :: SetTau( const double& t )
{
   initial_tau = t;
}
//--------------------------------------------------------------------------
void msdfSussmanFatemi :: SetFinalTime( const double& t )
{
   final_time = t;
}
//--------------------------------------------------------------------------
void msdfSussmanFatemi :: SetMersonAdaptivity( const double& t )
{
   merson_adaptivity = t;
}
//--------------------------------------------------------------------------
void msdfSussmanFatemi :: SetVerbosity( const int v )
{
   verbose = v;
   if( explicit_solver )
      explicit_solver -> SetVerbosity( verbose );
}
//--------------------------------------------------------------------------
void msdfSussmanFatemi :: GetExplicitRHS( const double& time,
                                          mGrid2D< double >& _u,
                                          mGrid2D< double >& _fu )
{
   const long int x_size = _u. GetXSize();
   const long int y_size = _u. GetYSize();
   const double& hx = _u. GetHx();
   const double& hy = _u. GetHy();
   const double _hx = 1.0 / hx;
   const double _hy = 1.0 / hy;
   const double* u = _u. Data();
   const double* u0 = _u0 -> Data();
   double* fu = _fu. Data();
   double* L = _l -> Data();
   double* heaviside_prime = _heaviside_prime -> Data();
   double* g2 = _g2 -> Data();
   //double* sign_u0 = _sign_u0 -> Data();
   const double* mod_grad_u0 = _mod_grad_u0 -> Data();
   const double delta_x = Max( hx, hy );
      
   long int i, j;
#ifdef HAVE_OPENMP
#pragma omp parallel for private( i, j ), firstprivate( _hx, _hy, u, heaviside_prime, L )
#endif
   for( i = 0; i < x_size; i ++ )
      for( j = 0; j < y_size; j ++ )
      {
         const long int ij = i * y_size + j;
         const double& u_ij = u[ ij ];

         double Dxf( 0.0 ), Dxb( 0.0 ), Dyf( 0.0 ), Dyb( 0.0 );
         if( i > 0 ) Dxb = ( u_ij - u[ ij - y_size ] ) * _hx;
         if( i < x_size - 1 ) Dxf = ( u[ ij + y_size ] - u_ij ) * _hx;
         if( i == 0 ) Dxb = Dxf;
         if( i == x_size - 1 ) Dxf = Dxb;
         
         if( j > 0 ) Dyb = ( u_ij - u[ ij - 1 ] ) * _hy;
         if( j < y_size - 1 ) Dyf = ( u[ ij + 1 ] - u_ij ) * _hy;
         if( j == 0 ) Dyb = Dyf;
         if( j == y_size - 1 ) Dyf = Dyb;
         
         const double sign_u0 = ( *_sign_u0 )( i, j );
         double Dx( 0.0 ), Dy( 0.0 );
         if( Dxf * sign_u0 < 0.0 && ( Dxf + Dxb ) * sign_u0 < 0.0 ) Dx = Dxf;           
         if( Dxb * sign_u0 > 0.0 && ( Dxf + Dxb ) * sign_u0 > 0.0 ) Dx = Dxb;           
         
         if( Dyf * sign_u0 < 0.0 && ( Dyf + Dyb ) * sign_u0 < 0.0 ) Dy = Dyf;           
         if( Dyb * sign_u0 > 0.0 && ( Dyf + Dyb ) * sign_u0 > 0.0 ) Dy = Dyb;           

         const double mod_grad_u = sqrt( Dx * Dx + Dy * Dy );
         L[ ij ] = sign_u0 * ( 1.0 - mod_grad_u );
      }

   const double int_coeffs[ 3 ][ 3 ] = { { 1.0, 1.0, 1.0 },
                                         { 1.0, 16.0, 1.0 },
                                         { 1.0, 1.0, 1.0 } };
   double max_lambda( 0.0 ), min_g2( DBL_MAX ), max_g1( 0.0 );
   const double tau = explicit_solver -> GetTau();
   //mGrid2D< double > lambda( _u ), _g1( _u ), _g2( _u );
   //mGrid2D< double > u_diff( _u );
#ifdef HAVE_OPENMP
#pragma omp parallel for private( i, j ), firstprivate( _hx, _hy, u, heaviside_prime, L, int_coeffs, mod_grad_u0 )
#endif
   for( i = 0; i < x_size; i ++ )
      for( j = 0; j < y_size; j ++ )
      {
         const long int ij = i * y_size + j;
         double lambda_ij( 0.0 );
         if( i > 0 && i < x_size - 1 && j > 0 && j < y_size - 1 )
         {
            double g1( 0.0 );
            long int m, n;
            for( m = -1; m <= 1; m ++ )
               for( n = -1; n <= 1; n ++ )
               {
                  double hv = ( * _heaviside_prime )( i + m, j + n );
                  g1 += int_coeffs[ m + 1 ][ n + 1 ] * hv * 
                     ( ( ( _u )( i + m, j + n ) - 
                         ( *_u0 )( i + m, j + n ) ) / tau + ( * _l )( i + m, j + n ) );
                  //g2 += int_coeffs[ m + 1 ][ n + 1 ] * hv * hv * ( * _mod_grad_u0 )( i + m, j + n );
               }
            if( g2[ ij ] != 0.0 ) lambda_ij = - g1 / g2[ ij ];
            //if( g2 != 0.0 && fabs( g2 ) < min_g2 ) min_g2 = fabs( g2 );
            //if( fabs( g1 ) > max_g1 ) max_g1 = fabs( g1 );
            //_g1( i, j ) = g1;
            //_g2( i, j ) = g2;
            //u_diff( i, j ) = ( ( _u )( i, j ) - ( *_u0 )( i, j ) ) * ( * _heaviside_prime )( i, j );
         }
         fu[ ij ] = L[ ij ] + lambda_ij * heaviside_prime[ ij ] * mod_grad_u0[ ij ];
         if( fabs( lambda_ij ) > max_lambda ) max_lambda = fabs( lambda_ij );
         //lambda( i, j ) = lambda_ij;
      }
   /*cout << endl << "\r Max. lambda_ij = " << setw( 15 ) << max_lambda << " min. g2 = " << setw( 15 ) << min_g2 << " max. g1 = " << setw( 15 ) << max_g1 << "         \r " << endl << flush;
   Draw( _u, "u", "gnuplot" );
   Draw( *_L, "L", "gnuplot" );
   Draw( _g1, "g1", "gnuplot" );
   Draw( _g2, "g2", "gnuplot" );
   Draw( u_diff, "u-diff", "gnuplot" );
   Draw( lambda, "lambda", "gnuplot" );
   Draw( _fu, "f", "gnuplot" );*/
   //getchar();*/
}
//--------------------------------------------------------------------------
msdfSussmanFatemi :: ~msdfSussmanFatemi()
{
   if( _u0 ) delete _u0;
   if( _heaviside_prime ) delete _heaviside_prime;
   if( _l ) delete _l;
   if( _mod_grad_u0 ) delete _mod_grad_u0;
}
