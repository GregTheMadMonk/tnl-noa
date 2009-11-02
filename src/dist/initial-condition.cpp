/***************************************************************************
                          initial-condition.cpp  -  description
                             -------------------
    begin                : 2007/06/17
    copyright            : (C) 2007 by Tomas Oberhuber
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

#include <math.h>
#include <mdist.h>
#include "initial-condition.h"

void Batman(  const double& t, void* crv_data, double& x, double& y )
{
   y = cos( t );
   x = 0.5 * sin( t ) + sin( y ) + sin( t ) * ( 0.2 + sin( t ) * sin( 3.0 * t ) * sin( 3.0 * t ) );
}

void ParametricCircle( const double& t, void* crv_data, double& x, double& y )
{
   const CircleData& c_data = * ( CircleData* ) crv_data;
   const double radius = c_data. radius;
   const double x_pos = c_data. x_pos;
   const double y_pos = c_data. y_pos;
   x = radius * cos( t ) + x_pos;
   y = radius * sin( t ) + y_pos;
}

void ParametricEllipse( const double& t, void* crv_data, double& x, double& y )
{
   const mVector< 2, double >& radius = * ( mVector< 2, double >* ) crv_data;
   x = radius[ 0 ] * cos( -t );
   y = radius[ 1 ] * sin( -t );
}

void ParametricPiskota( const double& t, void* crv_data, double& x, double& y )
{
   double _t = 2.0 * M_PI - t;
   x = 2.0 * pow( cos( _t ), 1.0 );
   y = 2.0 * pow( sin( _t ), 1.0 ) - 1.99 * pow( sin( _t ), 3.0 );
}

void ParametricSpiral( const double& t, void* crv_data, double& x, double& y )
{
   const SpiralData& s_data = * ( SpiralData* ) crv_data;
   x = s_data. radius * ( 0.5 * exp( -1.0 - 0.5 * sin( t ) ) - 0.025 * cos( t ) ) * cos( s_data. twist * atan( 1.0 + 0.5 * sin( t ) ) );
   y = s_data. radius * ( 0.5 * exp( -1.0 - 0.5 * sin( t ) ) - 0.025 * cos( t ) ) * sin( s_data. twist * atan( 1.0 + 0.5 * sin( t ) ) );
}
//--------------------------------------------------------------------------
bool GetInitialCondition( const mParameterContainer& parameters,
                          mGrid2D< double >*& _u )
{
   const double& radius = parameters. GetParameter< double >( "radius" );
   const double& amplitude = parameters. GetParameter< double >( "amplitude" );
   const double& frequency = parameters. GetParameter< double >( "frequency" );
   const double& power = parameters. GetParameter< double >( "power" );
   const double& sigma = parameters. GetParameter< double >( "sigma" );
   const double& shift = parameters. GetParameter< double >( "shift" );
   const double& phase = parameters. GetParameter< double >( "phase" );
   const double& x_position = parameters. GetParameter< double >( "x-position" );
   const double& y_position = parameters. GetParameter< double >( "y-position" );

   const mString& initial_condition = 
   parameters. GetParameter< mString >( "initial-condition" );

   assert( _u );
   mGrid2D< double >& u = * _u;
   const long int x_size = u. GetXSize();
   const long int y_size = u. GetYSize();
   const double& ax = u. GetAx();
   const double& ay = u. GetAy();
   const double& bx = u. GetBx();
   const double& by = u. GetBy();
   const double& hx = u. GetHx();
   const double& hy = u. GetHy();
   long int i, j;

   if( initial_condition == "sign-hole" )
   {
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            const double x = ax + i * hx;
            const double y = ay + j * hy;
            u( i, j ) = amplitude * ( Sign( x * x + y * y - radius ) - 1.0 );
         }
      return true;
   }
   if( initial_condition == "wavy-hole" )
   {
      const double rad2 = radius * radius;
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            const double x = ax + i * hx;
            const double y = ay + j * hy;
            u( i, j ) = amplitude * sin( phase * M_PI + frequency * M_PI * tanh( 5.0 * ( ( x * x + y * y ) - rad2 ) ) );
         }
      return true;
   }
   if( initial_condition == "wavy-hole-2" )
   {
      const double xi = 0.1;
      const double alpha = 10;
      const double theta_0 = 2;
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            const double x = ax + i * hx;
            const double y = ay + j * hy;
            const double r = sqrt( x * x + y * y );
            const double sign_z = Sign( r - radius );
            u( i, j ) = xi * sin( alpha * ( r - radius ) )
                        * exp( ( - alpha * alpha * pow( r - radius, 2.0 ) ) * ( theta_0 + sign_z ) );
         }
      return true;
   }
   if( initial_condition == "brigitte" )
   {
      const double m_fold = 2.0;
      const double r_0 = 0.6;
      const double theta_0 = 2.0;
      const double L_1 = 2.0;
      const double L_2 = 2.0;
      const double alpha = 10.0;
      const double y_0 = 2.0;
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            const double x = ax + i * hx;
            const double y = ay + j * hy;
            const double r = sqrt( x * x + y * y );
            u( i, j ) = -pow( sin( theta_0 * M_PI * x / L_1 ), m_fold ) * 
            ( 1.0 - 4.0 * pow( ( y - y_0 ) / L_2, 2.0 ) ) *
            ( 1.0 - tanh( alpha * ( r - radius ) ) ) / 2.0;
         }
      return true;
   }
   if( initial_condition == "exp-square" )
   {
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            const double x = ax + i * hx;
            const double y = ay + j * hy;
            const double n = power;
            const double c = sigma;
            const double BND = pow( radius, n );
            const double _BND = 1.0 / ( BND * BND );
            const double xn = pow( x, n );
            const double yn = pow( y, n );
            const double expcxy = exp( - c * ( x * x + y * y ) );
            const double f = _BND * ( xn - BND ) * ( yn - BND ) * expcxy;
            u( i, j ) = f;
         }
      return true;
   }
   if( initial_condition == "sin-waves" )
   {
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            const double x = ax + i * hx;
            const double y = ay + j * hy;
            u( i, j ) = amplitude * sin( frequency * M_PI * x ) * sin( frequency * M_PI * y );
         }
      return true;
   }
   if( initial_condition == "sin-waves-neumann" )
   {
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            const double x = ax + i * hx;
            const double y = ay + j * hy;
            u( i, j ) = amplitude * sin( frequency * M_PI * x ) * sin( frequency * M_PI * y );
         }
      return true;
   }
   if( initial_condition == "wavy-circles" )
   {
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            const double x = ax + i * hx;
            const double y = ay + j * hy;
            u( i, j ) = amplitude * sin( phase * M_PI + frequency * M_PI * sqrt( x * x + y * y ) );
         }
      return true;
   }
   if( initial_condition == "circle" )
   {
      const double& radius = parameters. GetParameter< double >( "radius" );
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            const double x = ax + i * hx;
            const double y = ay + j * hy;
            u( i, j ) = x * x + y * y - radius * radius;
         }
      return true;
   }
   if( initial_condition == "ellipse" )
   {
      const double& radius1 = parameters. GetParameter< double >( "radius1" );
      const double& radius2 = parameters. GetParameter< double >( "radius2" );
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            const double x = ( ax + i * hx ) / radius1;
            const double y = ( ay + j * hy ) / radius2;
            u( i, j ) = sqrt( x * x + y * y ) - 1.0;
         }
      return true;
   }
   if( initial_condition == "ellipse-circle-test" )
   {
      const double& radius1 = parameters. GetParameter< double >( "radius1" );
      const double& radius2 = parameters. GetParameter< double >( "radius2" );
      double ellipse_data[ 2 ] = { radius1, radius2 };
      mVector< 2, double > ellipse_radius( ellipse_data );
      CircleData circle_data;
      circle_data. radius = radius;
      circle_data. x_pos = x_position;
      circle_data. y_pos = y_position;
      return true;
   }
   if( initial_condition == "square" )
   {
      const double& radius = parameters. GetParameter< double >( "radius" );
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            const double x = ax + i * hx;
            const double y = ay + j * hy;
            u( i, j ) = Max( fabs( x ), fabs( y ) ) - radius;
         }
      return true;
   }
   if( initial_condition == "circle-square" )
   {
      const double& radius1 = parameters. GetParameter< double >( "radius1" );
      const double& radius2 = parameters. GetParameter< double >( "radius2" );
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            const double x = ax + i * hx;
            const double y = ay + j * hy;
            u( i, j ) = Min( sqrt( x * x + y * y ) - radius1 ,
                             -( Max( fabs( x ), fabs( y ) ) - radius2 ) );
         }
      return true;
   }
   if( initial_condition == "astroid" )
   {
      const double& radius = parameters. GetParameter< double >( "radius" );
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            const double x = ax + i * hx;
            const double y = ay + j * hy;
            u( i, j ) = pow( x * x, 1.0 / 3.0 ) +
                        pow( y * y, 1.0 / 3.0 ) -
                        pow( radius * radius, 1.0 / 3.0 );
         }
      return true;
   }
   if( initial_condition == "two-circles" )
   {
      const double& radius = parameters. GetParameter< double >( "radius" );
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            const double x = ax + i * hx;
            const double y = ay + j * hy;
            const double x1 = x - 0.5;
            const double x2 = x + 0.5;
            const double v1 = sqrt( x1 * x1 + y * y ) - radius;
            const double v2 = sqrt( x2 * x2 + y * y ) - radius;
            u( i, j ) = Min( v1, v2 );
         }
      return true;
   }
   if( initial_condition == "four-circles" )
   {
      const double& radius = parameters. GetParameter< double >( "radius" );
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            const double x = ax + i * hx;
            const double y = ay + j * hy;
            const double x1 = x - 0.5;
            const double x2 = x + 0.5;
            const double y1 = y - 0.5;
            const double y2 = y + 0.5;
            const double v1 = sqrt( x1 * x1 + y1 * y1 ) - radius;
            const double v2 = sqrt( x2 * x2 + y1 * y1 ) - radius;
            const double v3 = sqrt( x1 * x1 + y2 * y2 ) - radius;
            const double v4 = sqrt( x2 * x2 + y2 * y2 ) - radius;
            u( i, j ) = Min( v1, Min( v2, Min( v3, v4 ) ) );
         }
      return true;
   }
   if( initial_condition == "five-circles" )
   {
      const double& radius = parameters. GetParameter< double >( "radius" );
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            const double x = ax + i * hx;
            const double y = ay + j * hy;
            const double x1 = x - 1.5;
            const double x2 = x + 1.5;
            const double y1 = y - 0.25;
            const double y2 = y + 0.25;
            const double v1 = sqrt( x1 * x1 + y1 * y1 ) - radius;
            const double v2 = sqrt( x2 * x2 + y1 * y1 ) - radius;
            const double v3 = sqrt( x1 * x1 + y2 * y2 ) - radius;
            const double v4 = sqrt( x2 * x2 + y2 * y2 ) - radius;
            const double v5 = sqrt( x * x + y * y ) - radius;
            u( i, j ) = Min( v1, Min( v2, Min( v3, Min( v4, v5 ) ) ) );
         }
      return true;
   }
   if( initial_condition == "michal" )
   {
      const double& radius = parameters. GetParameter< double >( "radius" );
      const double& amplitude = parameters. GetParameter< double >( "amplitude" );
      const double& frequency = parameters. GetParameter< double >( "frequency" );
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            const double x = ax + i * hx;
            const double y = ay + j * hy;
            const double norm = sqrt( x * x + y * y );
            const double angl = acos( x / norm ) - M_PI_2;
            const double f = ( radius - amplitude * cos( frequency * angl ) * cos( frequency * angl ) ) * cos( angl );
            const double g = ( radius - amplitude * cos( frequency * angl ) * cos( frequency * angl ) ) * sin( angl );
            u( i, j ) = norm - sqrt( f * f + g * g );
         }
      return true;
   }
   if( initial_condition == "flower" )
   {
      const double& radius = parameters. GetParameter< double >( "radius" );
      const double& amplitude = parameters. GetParameter< double >( "amplitude" );
      const double& frequency = parameters. GetParameter< double >( "frequency" );
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            const double x = ax + i * hx;
            const double y = ay + j * hy;
            const double norm = sqrt( x * x + y * y );
            const double angl = acos( x / norm );
            const double r = radius + amplitude * sin( frequency * angl );
            u( i, j ) = norm - r;
         }
      return true;
   }
   if( initial_condition == "star" )
   {
      const double& radius = parameters. GetParameter< double >( "radius" );
      const double& amplitude = parameters. GetParameter< double >( "amplitude" );
      const double& frequency = parameters. GetParameter< double >( "frequency" );
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            const double x = ax + i * hx;
            const double y = ay + j * hy;
            const double norm = sqrt( x * x + y * y );
            const double angl = acos( x / norm );
            const double r = radius + amplitude * fabs( sin( frequency * angl ) );
            u( i, j ) = tanh( norm - r );
         }
      return true;
   }
   if( initial_condition == "piskota" )
   {
      GetParametrisedCurveSDF( u, 
                               ParametricPiskota,
                               NULL,
                               0, 6.28,
                               24, 3.0 );
      return true;
   }
   if( initial_condition == "spiral" )
   {
      SpiralData spiral_data;
      spiral_data. radius = parameters. GetParameter< double >( "radius" );
      spiral_data. twist = parameters. GetParameter< double >( "twist" );
      //mLevelSetCreator lsc;
      GetParametrisedCurveSDF( u, 
                               ParametricSpiral,
                               &spiral_data,
                               0, 6.28,
                               128, 3.0 );
      return true;
   }
   cerr << "Unknown initial condition '" << initial_condition << "' for the level-set formulation." << endl;
   return false;
}

