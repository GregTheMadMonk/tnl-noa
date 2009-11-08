/***************************************************************************
                          direct.cpp  -  description
                             -------------------
    begin                : 2005/08/10
    copyright            : (C) 2005 by Tom� Oberhuber
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

#include "../diff/mdiff.h"
#include "debug.h"
#include "direct.h"

//--------------------------------------------------------------------------
static double SolvePoint( const double& a,
                          const double& b,
                          const double& h,
                          const double& smallest )
{
   DBG_FUNCTION_NAME( "", "SolvePoint" );
   DBG_EXPR( a );
   DBG_EXPR( b );
   DBG_EXPR( h );
 
   double sgn = Sign( smallest );
   DBG_EXPR( smallest - a );
   DBG_EXPR( smallest - b );
   if( fabs( smallest - a ) >= h || fabs( smallest - b ) >= h )
   {
      return  smallest + sgn * h ;
   }
   //if( a == 0.0 ) 
   //   return b + sign * h;
   if( b == 0.0 )
   {
      return a + sgn * h;
   }
   double B = - 2.0 * ( a + b );
   double C = a * a + b * b - h * h;
   double D = B * B - 8.0 * C;
   DBG_EXPR( D );
   
   assert( D >= 0.0 );
   double x1 = ( -B + sqrt( D ) ) / 4.0;
   double x2 = ( -B - sqrt( D ) ) / 4.0;
   if( sgn * x1 < sgn * x2 )
      return x2;
   return x1;
}
//--------------------------------------------------------------------------
static double CountRoot( const double& a, const double& b )
{
   assert( a * b < 0.0 );
   return -1.0 * a / ( b - a );
}
//--------------------------------------------------------------------------
double UpdatePoint2D( const mGrid2D< double >& f,
                      int i, int j,
                      const double& smallest,
                      mDstDirection smallest_direction,
                      mDstState e,
                      mDstState w,
                      mDstState s,
                      mDstState n )
{
   //cout << "Hx = " << f. GetHx() << " Hy = " << f. GetHy() << endl;
   assert( f. GetHx() == f. GetHy() );
   double h = f. GetHx();
   if( smallest_direction == mDstEast )
   {
      if( s == mDstFar && n == mDstFar )
         return SolvePoint( f( i + 1, j ), 0.0, h, smallest );
      if( s != mDstFar && n != mDstFar )
         if( fabs( f( i, j + 1 ) ) < fabs( f( i, j - 1 ) ) )
            return SolvePoint( f( i + 1, j ), f( i, j + 1 ), h, smallest ) ;
         else return SolvePoint( f( i + 1, j ), f( i, j - 1 ), h, smallest );
      if( s != mDstFar )
         return SolvePoint( f( i + 1, j ), f( i, j - 1 ), h, smallest );
      return SolvePoint( f( i + 1, j ), f( i, j + 1 ), h, smallest );
   }

   if( smallest_direction == mDstWest )
   {
      if( s == mDstFar && n == mDstFar )
         return SolvePoint( f( i - 1, j ), 0.0, h, smallest );
      if( s != mDstFar && n != mDstFar )
         if( fabs( f( i, j + 1 ) ) < fabs( f( i, j - 1 ) ) )
            return SolvePoint( f( i - 1, j ), f( i, j + 1 ), h, smallest ) ;
         else return SolvePoint( f( i - 1, j ), f( i, j - 1 ), h, smallest );
      if( s != mDstFar )
         return SolvePoint( f( i - 1, j ), f( i, j - 1 ), h, smallest );
      return SolvePoint( f( i - 1, j ), f( i, j + 1 ), h, smallest );
   }

   if( smallest_direction == mDstNorth )
   {
      if( w == mDstFar && e == mDstFar )
         return SolvePoint( f( i, j + 1 ), 0.0, h, smallest );
      if( w != mDstFar && e != mDstFar )
         if( fabs( f( i + 1, j ) ) < fabs( f( i - 1, j ) ) )
            return SolvePoint( f( i, j + 1 ), f( i + 1, j ), h, smallest ) ;
         else return SolvePoint( f( i, j + 1 ), f( i - 1, j ), h, smallest );
      if( w != mDstFar )
         return SolvePoint( f( i, j + 1 ), f( i - 1, j ), h, smallest );
      return SolvePoint( f( i, j + 1 ), f( i + 1, j ), h, smallest );
   }
   
   if( smallest_direction == mDstSouth )
   {
      if( w == mDstFar && e == mDstFar )
         return SolvePoint( f( i, j - 1 ), 0.0, h, smallest );
      if( w != mDstFar && e != mDstFar )
         if( fabs( f( i + 1, j ) ) < fabs( f( i - 1, j ) ) )
            return SolvePoint( f( i, j - 1 ), f( i + 1, j ), h, smallest ) ;
         else return SolvePoint( f( i, j - 1 ), f( i - 1, j ), h, smallest );
      if( w != mDstFar )
         return SolvePoint( f( i, j - 1 ), f( i - 1, j ), h, smallest );
      return SolvePoint( f( i, j - 1 ), f( i + 1, j ), h, smallest );
   }
   assert( 0 );
   return 0.0;
}

