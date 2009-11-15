/***************************************************************************
                          mdist-test-param-crv-dst.cpp  -  description
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

//--------------------------------------------------------------------------
#include <diff/mdiff.h>
#include "debug.h"
#include "mdist.h"

void ParamCircle( const double& t, void* crv_data, double& x, double& y )
{
   double rad = * ( double* ) crv_data;
   x = rad * sin( t );
   y = rad * cos( t );
}

void Batman(  const double& t, void* crv_data, double& x, double& y )
{
   x = cos( t );
   y = 0.5 * sin( t ) + sin( x ) + sin( t ) * ( 0.2 + sin( t ) * sin( 3.0 * t ) * sin( 3.0 * t ) );
}

int main( int argc, char* argv[] )
{
   dbgFunctionName( "", "main" );
   dbgInit( "debug.xml" );

   const int N = 100;
   mGrid2D< double > u( N, N, -2.0, 2.0, -2.0, 2.0 );
   mCurve< mVector< 2, double > > curve;
   const double& hx = u. GetHx();
   const double& hy = u. GetHy();
   const double radius = 1.0;

   cout << "Restoring the SDF of a circle..." << endl;
   long int i, j;
   for( i = 0; i < N; i ++ )
      for( j = 0; j < N; j ++ )
      {
         const double x = -2.0 + i * hx;
         const double y = -2.0 + j * hy;
         u( i, j ) = x * x + y * y - radius * radius;
      }
   cout << "Writing original circle level-set function in circle-orig..." << endl;
   Draw( u, "circle-orig", "gnuplot" );
   GetLevelSetCurve( u, curve );
   DstFastSweeping2D( u, 4 );
   GetLevelSetCurve( u, curve );
   cout << "Writing circle SDF level-set function in circle-dst..." << endl;
   Draw( u, "circle-dst", "gnuplot" );
   cout << "Writing curve circle-crv..." << endl;
   Write( curve, "circle-crv", "gnuplot" );
   curve. Erase();

   cout << "Restoring the SDF of a ellipse..." << endl;
   for( i = 0; i < N; i ++ )
      for( j = 0; j < N; j ++ )
      {
         const double x = -2.0 + i * hx;
         const double y = -2.0 + j * hy;
         u( i, j ) = 0.25 * x * x + y * y - radius * radius;
      }
   cout << "Writing original ellipse level-set function in ellipse-orig..." << endl;
   Draw( u, "ellipse-orig", "gnuplot" );
   GetLevelSetCurve( u, curve );
   DstFastSweeping2D( u, 4 );
   GetLevelSetCurve( u, curve );
   cout << "Writing ellipse SDF level-set function in ellipse-dst..." << endl;
   Draw( u, "ellipse-dst", "gnuplot" );
   cout << "Writing curve ellipse-crv..." << endl;
   Write( curve, "ellipse-crv", "gnuplot" );
   curve. Erase();

   cout << "Restoring the SDF of an astroid..." << endl;
   for( i = 0; i < N; i ++ )
      for( j = 0; j < N; j ++ )
      {
         const double x = -2.0 + i * hx;
         const double y = -2.0 + j * hy;
         u( i, j ) = pow( x * x, 1.0 / 3.0 ) +
                     pow( y * y, 1.0 / 3.0 ) - 
                     pow( radius * radius, 1.0 / 3.0 );
      }
   cout << "Writing original astroid level-set function in astroid-orig..." << endl;
   Draw( u, "astroid-orig", "gnuplot" );
   GetLevelSetCurve( u, curve );
   DstFastSweeping2D( u, 4 );
   GetLevelSetCurve( u, curve );
   cout << "Writing astroid SDF level-set function in astroid-dst..." << endl;
   Draw( u, "astroid-dst", "gnuplot" );
   cout << "Writing curve astroid-crv..." << endl;
   Write( curve, "astroid-crv", "gnuplot" );
   curve. Erase();


   
   cout <<  "Getting the curve SDF..." << flush << endl;
   GetParametrisedCurveSDF( u, 
                            ParamCircle,
                            ( void* ) &radius,
                            0, 6.28, 4,
                            3.0 );
   
   Draw( u, "circle-dst", "gnuplot" );
   
   GetParametrisedCurveSDF( u, 
                            Batman,
                            NULL,
                            0, 6.28, 24,
                            3.0 );
   Draw( u, "batman-dst", "gnuplot" );
   GetLevelSetCurve( u, curve );
   cout << "Writing curve batman-crv..." << endl;
   Write( curve, "batman-crv", "gnuplot" );
}
