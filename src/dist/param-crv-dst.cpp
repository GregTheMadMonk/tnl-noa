/***************************************************************************
                          param-crv-dst.cpp  -  description
                             -------------------
    begin                : 2007/02/24
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

#include "param-crv-dst.h"

#include <math.h>
#include <float.h>

#include <debug/tnlDebug.h>
#include <core/mfuncs.h>
#include "fsm2d.h"


//--------------------------------------------------------------------------
mLevelSetCreator :: mLevelSetCreator()
   : fixed_points( 0 )
{
}
//--------------------------------------------------------------------------
mLevelSetCreator :: ~mLevelSetCreator()
{
   if( fixed_points ) delete fixed_points;
}
//--------------------------------------------------------------------------
bool mLevelSetCreator :: Init( mGrid2D< double >& u )
{
   const int x_size = u. GetXSize();
   const int y_size = u. GetYSize();

   if( fixed_points )
   {
      if( fixed_points -> GetXSize() != x_size ||
          fixed_points -> GetYSize() != y_size )
      {
         delete fixed_points;
         fixed_points = new mField2D< bool >( x_size, y_size );
      }
   }
   else fixed_points = new mField2D< bool >( x_size, y_size );
   fixed_points -> Zeros();

   const double& h_x = u. GetHx();
   const double& h_y = u. GetHy();
   double h = sqrt( h_x * h_x + h_y * h_y );

   long int i, j;
   for( i = 0; i < x_size; i ++ )
      for( j = 0; j < y_size; j ++ )
         u( i, j ) = DBL_MAX;
   return true;
}
//--------------------------------------------------------------------------
void mLevelSetCreator :: DrawCurve( mGrid2D< double >& u,
                                    void ( *crv )( const double& t, void* crv_data, double& pos_x, double& pos_y ),
                                    void* crv_data,
                                    const double& t1,
                                    const double& t2,
                                    const double quality )
{
   dbgFunctionName( "mLevelSetCreator", "DrawCurve" );

   const double& h_x = u. GetHx();
   const double& h_y = u. GetHy();
   const double& a_x = u. GetAx();
   const double& a_y = u. GetAy();
   double h = sqrt( h_x * h_x + h_y * h_y );
   const long int x_size = u. GetXSize();
   const long int y_size = u. GetYSize();

   double t( t1 );
   double tau = ( t2 - t1 ) / quality;
   double last_x, last_y;
   crv( t, crv_data, last_x, last_y );
   last_x -= a_x;
   last_y -= a_y;
   while( t <= t2 )
   {
      //cout << " t = " << t << endl;
      //cout << " tau = " << tau << endl;
      double x, y;
      crv( t + tau, crv_data, x, y );
      x -= a_x;
      y -= a_y;
      double x_dir = x - last_x;
      double y_dir = y - last_y;
      double point_dist = sqrt( x_dir * x_dir + y_dir * y_dir );
      //cout << "point_dist = " << point_dist << endl;
      if( point_dist > h / quality )
      {
         tau /= 2; // quality * h / point_dist ;
         continue;
      }
      long int i = ( long int ) ( x / h_x );
      long int j = ( long int ) ( y / h_y );
      //cout << "x_dir = "<< x_dir << "y_dir = " << y_dir << endl;
      double dist, vx, vy;
      if( i >= 0 && j >= 0 && i < x_size && j < y_size )
      {
         vx = i * h_x;
         vy = j * h_y;
         double vx_dir = vx - x;
         double vy_dir = vy - y;
         dist = sqrt( vx_dir * vx_dir + vy_dir * vy_dir );
         double vsign = Sign( y_dir * vx_dir - x_dir * vy_dir );
         //cout << "sign:" << vsign << endl;
         if( fabs( u( i, j ) ) > dist )
            u( i, j ) = -vsign * dist;
         ( *fixed_points )( i, j ) = true;
      }
      if( i + 1 >= 0 && j >= 0 && i + 1 < x_size && j < y_size )
      {
         vx = ( i + 1 ) * h_x;
         vy = j * h_y;
         double vx_dir = vx - x;
         double vy_dir = vy - y;
         dist = sqrt( vx_dir * vx_dir + vy_dir * vy_dir );
         double vsign = Sign( y_dir * vx_dir - x_dir * vy_dir );
         //cout << "sign:" << vsign << endl;
         if( fabs( u( i + 1, j ) ) > dist )
            u( i + 1, j ) = -vsign * dist;
         ( *fixed_points )( i + 1, j ) = true;
      }
      if( i >= 0 && j + 1 >= 0 && i < x_size && j + 1 < y_size )
      {
         vx = i * h_x;
         vy = ( j + 1 ) * h_y;
         double vx_dir = vx - x;
         double vy_dir = vy - y;
         dist = sqrt( vx_dir * vx_dir + vy_dir * vy_dir );
         double vsign = Sign( y_dir * vx_dir - x_dir * vy_dir );
         //cout << "sign:" << vsign << endl;
         if( fabs( u( i, j + 1 ) ) > dist )
            u( i, j + 1 ) = -vsign * dist;
         ( *fixed_points )( i, j + 1 ) = true;
      }
      if( i + 1 >= 0 && j + 1 >= 0 && i + 1 < x_size && j + 1 < y_size )
      {
         vx = ( i + 1 ) * h_x;
         vy = ( j + 1 ) * h_y;
         double vx_dir = vx - x;
         double vy_dir = vy - y;
         dist = sqrt( vx_dir * vx_dir + vy_dir * vy_dir );
         double vsign = Sign( y_dir * vx_dir - x_dir * vy_dir );
         //cout << "sign:" << vsign << endl;
         if( fabs( u( i + 1, j + 1 ) ) > dist )
            u( i + 1, j + 1 ) = -vsign * dist;
         ( *fixed_points )( i + 1, j + 1 ) = true;
      }
      if( i - 1 >= 0 && j >= 0 && i - 1 < x_size && j < y_size )
      {
         vx = ( i - 1 ) * h_x;
         vy = ( j ) * h_y;
         double vx_dir = vx - x;
         double vy_dir = vy - y;
         dist = sqrt( vx_dir * vx_dir + vy_dir * vy_dir );
         double vsign = Sign( y_dir * vx_dir - x_dir * vy_dir );
         //cout << "sign:" << vsign << endl;
         if( fabs( u( i - 1, j ) ) > dist )
            u( i - 1, j ) = -vsign * dist;
         ( *fixed_points )( i - 1, j ) = true;
      }
      if( i >= 0 && j - 1 >= 0 && i < x_size && j - 1 < y_size )
      {
         vx = ( i ) * h_x;
         vy = ( j - 1 ) * h_y;
         double vx_dir = vx - x;
         double vy_dir = vy - y;
         dist = sqrt( vx_dir * vx_dir + vy_dir * vy_dir );
         double vsign = Sign( y_dir * vx_dir - x_dir * vy_dir );
         //cout << "sign:" << vsign << endl;
         if( fabs( u( i, j - 1 ) ) > dist )
            u( i, j - 1 ) = -vsign * dist;
         ( *fixed_points )( i, j - 1 ) = true;
      }
      if( i - 1 >= 0 && j - 1 >= 0 && i - 1 < x_size && j - 1 < y_size )
      {
         vx = ( i - 1 ) * h_x;
         vy = ( j - 1 ) * h_y;
         double vx_dir = vx - x;
         double vy_dir = vy - y;
         dist = sqrt( vx_dir * vx_dir + vy_dir * vy_dir );
         double vsign = Sign( y_dir * vx_dir - x_dir * vy_dir );
         //cout << "sign:" << vsign << endl;
         if( fabs( u( i - 1, j - 1 ) ) > dist )
            u( i - 1, j - 1 ) = -vsign * dist;
         ( *fixed_points )( i - 1, j - 1 ) = true;
      }

      tau *= 1.5;
      t += tau;
      last_x = x;
      last_y = y;
   }
   /*fstream file;
   file. open( "fixed", ios :: out );
   for( i = 0; i < x_size; i ++ )
      for( j = 0; j < y_size; j ++ )
         file << i << " " << j << " " << fixed( i, j ) << endl;
   file. close();*/

   //u. DrawFunction( "u" );
}
//--------------------------------------------------------------------------
void mLevelSetCreator :: Finalize( mGrid2D< double >& u,
                                   const int sweepings )
{
   /* Detect the interior and the exterior using pseudo fast sweepind method.
    * For the points outside the fixed narrow band where we know the distance to the
    * curve we set the sign by the neighbours.
    */
   
   const long int x_size = u. GetXSize();
   const long int y_size = u. GetYSize();
   int s1, s2, i, j;
   /*for( i = 0; i < x_size; i ++ )
      for( j = 0; j < y_size; j ++ )
         if( */

   double max_val = Max( u. GetBx() - u. GetAx(), u. GetBy() - u. GetAy() );
   for( s1 = -1; s1 <= 1; s1 += 2 )
      for( s2 = -1; s2 <= 1; s2 += 2 )
      {
         for( i = ( s1 < 0 ? x_size - 1 : 0 ); ( s1 < 0 ? i >= 0 : i < x_size ); i += s1 )
            for( j = ( s2 < 0 ? y_size - 1 : 0 ); ( s2 < 0 ? j >= 0 : j < y_size ); j += s2 )
            {
               if( ( *fixed_points )( i, j ) == true ) continue;
               if( u( i, j ) != DBL_MAX ) continue;
               if( i > 0 && u( i - 1, j ) != DBL_MAX )
               {
                  u( i, j ) = Sign( u( i - 1, j ) ) * max_val;
                  continue;
               }
               if( i <  x_size - 1 && u( i + 1, j ) != DBL_MAX )
               {
                  u( i, j ) = Sign( u( i + 1, j ) ) * max_val;
                  continue;
               }
               if( j > 0 && u( i, j - 1 ) != DBL_MAX )
               {
                  u( i, j ) = Sign( u( i, j - 1 ) ) * max_val;
                  continue;
               }
               if( j <  y_size - 1 && u( i, j + 1 ) != DBL_MAX )
                  u( i, j ) = Sign( u( i, j + 1 ) ) * max_val;
            }
      }
   /*Draw( u, "phi", "gnuplot" );
   cout << "Enter..." << endl;
   getchar();*/
   DstFastSweeping2D( u, sweepings, fixed_points, false );
}
//--------------------------------------------------------------------------
void GetParametrisedCurveSDF( mGrid2D< double >& u, 
                              void ( *crv )( const double& t, void* crv_data, double& pos_x, double& pos_y ),
                              void* crv_data,
                              const double& t1,
                              const double& t2,
                              const int sweepings,
                              const double quality )
{
   dbgFunctionName( "", "GetParametrisedCurveSDF" );

   const double& h_x = u. GetHx();
   const double& h_y = u. GetHy();
   const double& a_x = u. GetAx();
   const double& a_y = u. GetAy();
   double h = sqrt( h_x * h_x + h_y * h_y );
   const int x_size = u. GetXSize();
   const int y_size = u. GetYSize();

   mField2D< bool > fixed( x_size, y_size );
   fixed. Zeros();

   int i, j;
   for( i = 0; i < x_size; i ++ )
      for( j = 0; j < y_size; j ++ )
         u( i, j ) = 5.0 * h;
   double t( t1 );
   double tau = ( t2 - t1 ) / quality;
   double last_x, last_y;
   crv( t, crv_data, last_x, last_y );
   last_x -= a_x;
   last_y -= a_y;
   while( t <= t2 )
   {
      //cout << " t = " << t << endl;
      //cout << " tau = " << tau << endl;
      double x, y;
      crv( t + tau, crv_data, x, y );
      x -= a_x;
      y -= a_y;
      double x_dir = x - last_x;
      double y_dir = y - last_y;
      double point_dist = sqrt( x_dir * x_dir + y_dir * y_dir );
      //cout << "point_dist = " << point_dist << endl;
      if( point_dist > h / quality )
      {
         tau /= 2; // quality * h / point_dist ;
         continue;
      }
      int i = ( int ) ( x / h_x );
      int j = ( int ) ( y / h_y );
      //cout << "x_dir = "<< x_dir << "y_dir = " << y_dir << endl;
      double dist, vx, vy;
      if( i >= 0 && j >= 0 && i < x_size && j < y_size )
      {
         vx = i * h_x;  
         vy = j * h_y;
         double vx_dir = vx - x;  
         double vy_dir = vy - y;  
         dist = sqrt( vx_dir * vx_dir + vy_dir * vy_dir );
         double vsign = Sign( y_dir * vx_dir - x_dir * vy_dir );
         //cout << "sign:" << vsign << endl;
         if( fabs( u( i, j ) ) > dist )
            u( i, j ) = -vsign * dist;
         fixed( i, j ) = true;
      }
      if( i + 1 >= 0 && j >= 0 && i + 1 < x_size && j < y_size )
      {
         vx = ( i + 1 ) * h_x;  
         vy = j * h_y;
         double vx_dir = vx - x;  
         double vy_dir = vy - y;  
         dist = sqrt( vx_dir * vx_dir + vy_dir * vy_dir );
         double vsign = Sign( y_dir * vx_dir - x_dir * vy_dir );
         //cout << "sign:" << vsign << endl;
         if( fabs( u( i + 1, j ) ) > dist )
            u( i + 1, j ) = -vsign * dist;
         fixed( i + 1, j ) = true;
      }
      if( i >= 0 && j + 1 >= 0 && i < x_size && j + 1 < y_size )
      {
         vx = i * h_x;  
         vy = ( j + 1 ) * h_y;
         double vx_dir = vx - x;  
         double vy_dir = vy - y;  
         dist = sqrt( vx_dir * vx_dir + vy_dir * vy_dir );
         double vsign = Sign( y_dir * vx_dir - x_dir * vy_dir );
         //cout << "sign:" << vsign << endl;
         if( fabs( u( i, j + 1 ) ) > dist )
            u( i, j + 1 ) = -vsign * dist;
         fixed( i, j + 1 ) = true;
      }
      if( i + 1 >= 0 && j + 1 >= 0 && i + 1 < x_size && j + 1 < y_size )
      {
         vx = ( i + 1 ) * h_x;  
         vy = ( j + 1 ) * h_y;
         double vx_dir = vx - x;  
         double vy_dir = vy - y;  
         dist = sqrt( vx_dir * vx_dir + vy_dir * vy_dir );
         double vsign = Sign( y_dir * vx_dir - x_dir * vy_dir );
         //cout << "sign:" << vsign << endl;
         if( fabs( u( i + 1, j + 1 ) ) > dist )
            u( i + 1, j + 1 ) = -vsign * dist;
         fixed( i + 1, j + 1 ) = true;
      }
      tau *= 1.5;
      t += tau;
      last_x = x;
      last_y = y;
   }
  
   /*fstream file;
   file. open( "fixed", ios :: out );
   for( i = 0; i < x_size; i ++ )
      for( j = 0; j < y_size; j ++ )
         file << i << " " << j << " " << fixed( i, j ) << endl;
   file. close();*/

   //u. DrawFunction( "u" );

   
   DstFastSweeping2D( u, sweepings, &fixed, false );

}
