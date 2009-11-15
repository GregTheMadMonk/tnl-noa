/***************************************************************************
                          fsm2d.cpp  -  description
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

#include <float.h>
#include "debug.h"
#include "direct.h"
#include "fsm2d.h"
#include "init_band.h"

//--------------------------------------------------------------------------
bool DstFastSweeping2D( mGrid2D< double >& phi, 
                        int sweepings,
                        mField2D< bool >* _fixed,
                        mGrid2D< double >* extend_velocity,
                        bool verbose )
{
   dbgFunctionName( "", "DstFastSweeping" );
   if( phi. GetHx() != phi. GetHy() )
   {
      cerr << "The FAST SWEEPING METHOD does not support different space steps along different axis. i.e. phi. GetHx() == phi. GetHy() must hold." << endl;
      return false;
   }
   int x_size = phi. GetXSize();
   int y_size = phi. GetYSize();

   const double h = phi. GetHx();
   const double h2 = h * h;
   
   mField2D< bool >* fixed;
   if( ! _fixed )
   {
      int stack_end( 0 );
      mVector< 2, int >* stack = new mVector< 2, int >[ x_size * y_size ];
      mField2D< char > in_stack( x_size, y_size );
      in_stack. Zeros();

      InitBand( phi, stack, stack_end, in_stack );  
      //RedistanceBand( phi, stack, stack_end, in_stack );

      fixed = new mField2D< bool >( x_size, y_size );
      fixed -> Zeros();

      int i;
      for( i = 0; i < stack_end; i ++ )
         ( *fixed )( stack[ i ][ 0 ], stack[ i ][ 1 ] ) = true;
      delete stack;
   }
   else
      fixed = _fixed;
      
   if( verbose )
     cout << "Size: " << phi. GetXSize() << "x" << phi. GetYSize() << endl;

   int iter( 0 );
   double err = DBL_MAX;
   int k;
   for( k = 0; k < sweepings; k ++ )
   {
      iter ++;
      int i, j, s1, s2;
      for( s1 = -1; s1 <= 1; s1 += 2 )
         for( s2 = -1; s2 <= 1; s2 += 2 )
         {
            
            /*Draw( phi, "phi", "gnuplot" );
            if( extend_velocity )
               Draw( *extend_velocity, "V", "gnuplot" );
            cout << "Enter..." << endl;
            getchar();*/

            for( i = ( s1 < 0 ? x_size - 1 : 0 ); ( s1 < 0 ? i >= 0 : i < x_size ); i += s1 )
               for( j = ( s2 < 0 ? y_size - 1 : 0 ); ( s2 < 0 ? j >= 0 : j < y_size ); j += s2 )
               {
                  err = 0.0;
                  dbgCout( "Updating point: " << i << " " << j );
                  double c = phi( i, j );
                  if( ( *fixed )( i, j ) == false )
                  {
                     long int u_xmin_i( i ), u_ymin_j( j );
                     double u_xmin( DBL_MAX ), u_ymin( DBL_MAX );
                     if( i == 0 )
                     {
                        u_xmin = phi( i + 1, j );
                        u_xmin_i ++;
                     }
                     else if( i == x_size - 1 )
                        {
                           u_xmin = phi( i - 1, j );
                           u_xmin_i --;
                        }
                        else if( fabs( phi( i - 1, j ) ) < fabs( phi( i + 1, j ) ) ) 
                              {
                                 u_xmin = phi( i - 1, j );
                                 u_xmin_i --;
                              }
                           else
                              {
                                 u_xmin = phi( i + 1, j );
                                 u_xmin_i ++;
                              }
                     
                     if( j == 0 )
                     {
                        u_ymin = phi( i, j + 1 );
                        u_ymin_j ++;
                     }
                     else if( j == y_size - 1 )
                        {
                           u_ymin = phi( i, j - 1 );
                           u_ymin_j --;
                        }
                        else if( fabs( phi( i, j - 1 ) ) < fabs( phi( i, j + 1 ) ) )
                           {
                              u_ymin = phi( i, j - 1 );
                              u_ymin_j --;
                           }
                           else
                           {
                              u_ymin = phi( i, j + 1 );
                              u_ymin_j ++;
                           }

                     assert( u_xmin != DBL_MAX && u_ymin != DBL_MAX );
                     if( c > 0 )
                     {
                        /* For c > 0 we solve the eqution:
                         *
                         * [ ( u_ij - u_xmin )^+ ]^2 + [ ( u_ij - u_ymin )^+ ]^2 = h^2, (1)
                         *
                         * where 
                         *
                         * u_xmin = Min( u_{i-1,j}, u_{i+1,j},
                         * u_ymin = Min( u_{i,j-1}, u_{i,j+1},
                         * (x)^+ = Max( x, 0 ).
                         *
                         * Write (1) in a form
                         *
                         * [ ( u_ij - a_1 )^+ ]^2 + [ ( u_ij - a_2 )^+ ]^2 = h^2, (2)
                         *
                         * and assume a_1 < a_2. Setting a3 = \infty than there
                         * exists integer p such that a_p < u_new <= a_{p+1} and u_new is 
                         * a solution of (2) because ( u_new - a_i )^+ = 0 for i > p;
                         * We start with p = 1:
                         * Set u_new = a_1 + h.
                         * If u_new <= a_2 we have the solution.
                         *
                         */
                        
                        const double a1 = Min( u_xmin, u_ymin );
                        const double a2 = Max( u_xmin, u_ymin );
                        double u_new = a1 + h;
                        if( u_new <= a2 )
                        {
                           phi( i, j ) = u_new;
                           if( extend_velocity )
                           {
                              if( a1 == u_xmin ) ( *extend_velocity )( i, j ) = ( *extend_velocity )( u_xmin_i, j );
                              else ( *extend_velocity )( i, j ) = ( *extend_velocity )( i, u_ymin_j );
                           }
                           continue;
                        }

                        /* Otherwise we solve the equation:
                         * ( u_ij - a_1 )^2 + ( u_ij - a_2 )^2 = h^2, (3)
                         *
                         * that is
                         *
                         * 2.0 * u_ij^2 - 2.0 * ( a_1 + a_2 ) * u_ij + a_1^2 + a_2^2 - h^2 = 0
                         *
                         */
                        
                        const double a = 2.0;
                        const double b = -2.0 * ( a1 + a2 );
                        const double c = a1 * a1 + a2 * a2  - h * h;
                        const double D = b * b - 4.0 * a * c;
                        assert( D >= 0.0 );
                        const double u1 = ( - b + sqrt( D ) ) / ( 2.0 * a );
                        const double u2 = ( - b - sqrt( D ) ) / ( 2.0 * a );
                        phi( i, j ) = Max( u1, u2 );
                        if( extend_velocity )
                        {
                           const double vx = ( *extend_velocity )( u_xmin_i, j );
                           const double vy = ( *extend_velocity )( i, u_ymin_j );
                           double dx = phi( i, j ) - u_xmin;
                           double dy = phi( i, j ) - u_ymin;
                           dx *= dx / h2;
                           dy *= dy / h2;
                           /* Now we have that dx + dy = 1. We will use it for wieghted approximation 
                            * of the normal speed extension.
                            */
                           ( *extend_velocity )( i, j ) = Max( vx, vy ); //( 1.0 - dx ) * vx + ( 1.0 - dy ) * vy;

                        }
                        
                     }
                     else
                     {
                        /* The equation (1) now turns to
                         *
                         * [ ( u_ij - u_xmin )^- ]^2 + [ ( u_ij - u_ymin )^- ]^2 = h^2, (4)
                         *
                         * where
                         *
                         * (x)^- = Min( x, 0 ).
                         *
                         * and we proceed in very similar way. We write (4) in a form
                         *
                         * [ ( u_ij - a_1 )^+ ]^2 + [ ( u_ij - a_2 )^+ ]^2 = h^2, (5)
                         *
                         * and assume a_1 > a_2 (a_1 are negative now). Setting a3 = -\infty than there
                         * exists integer p such that a_p > u_new >= a_{p+1} and u_new is 
                         * a solution of (5) because ( u_new - a_i )^- = 0 for i > p;
                         * We start with p = 1:
                         * Set u_new = a_1 - h.
                         * If u_new >= a_2 we have the solution.
                         */
                     
                        const double a1 = Max( u_xmin, u_ymin );
                        const double a2 = Min( u_xmin, u_ymin );
                        double u_new = a1 - h;
                        if( u_new >= a2 )
                        {
                           phi( i, j ) = u_new;
                           if( extend_velocity )
                           {
                              if( a1 == u_xmin ) ( *extend_velocity )( i, j ) = ( *extend_velocity )( u_xmin_i, j );
                              else ( *extend_velocity )( i, j ) = ( *extend_velocity )( i, u_ymin_j );
                           }
                           continue;
                        }

                        /* Otherwise solve the same equation as (3).
                         */
                        
                        const double a = 2.0;
                        const double b = -2.0 * ( a1 + a2 );
                        const double c = a1 * a1 + a2 * a2  - h * h;
                        const double D = b * b - 4.0 * a * c;
                        assert( D >= 0.0 );
                        const double u1 = ( - b + sqrt( D ) ) / ( 2.0 * a );
                        const double u2 = ( - b - sqrt( D ) ) / ( 2.0 * a );
                        phi( i, j ) = Min( u1, u2 );
                        if( extend_velocity )
                        {
                           const double vx = ( *extend_velocity )( u_xmin_i, j );
                           const double vy = ( *extend_velocity )( i, u_ymin_j );
                           double dx = phi( i, j ) - u_xmin;
                           double dy = phi( i, j ) - u_ymin;
                           dx *= dx / h2;
                           dy *= dy / h2;
                           /* Now we have that dx + dy = 1. We will use it for wieghted approximation 
                            * of the normal speed extension.
                            */
                           ( *extend_velocity )( i, j ) = Min( vx, vy ); //( 1.0 - dx ) * vx + ( 1.0 - dy ) * vy;

                        }
                     }


                     /*double smallest( DBL_MAX );
                     mDstDirection smallest_direction;
                     if( i < x_size - 1 && 
                         fabs( phi( i + 1, j ) ) < fabs( smallest ) )
                     {
                        smallest = phi( i + 1, j );
                        smallest_direction = mDstEast;
                     }
                     if( i > 0 && 
                         fabs( phi( i - 1, j ) ) < fabs( smallest ) )
                     {
                        smallest = phi( i - 1, j );
                        smallest_direction = mDstWest;
                     }
                     if( j < y_size - 1 && 
                         fabs( phi( i, j + 1 ) ) < fabs( smallest ) )
                     {
                        smallest = phi( i, j + 1 );
                        smallest_direction = mDstNorth;
                     }
                     if( j > 0 &&
                         fabs( phi( i, j - 1 ) ) < fabs( smallest ) )
                     {
                        smallest = phi( i, j - 1 );
                        smallest_direction = mDstSouth;
                     }
                     mDstState e( mDstTentative ),
                               w( mDstTentative ),
                               s( mDstTentative ),
                               n( mDstTentative );
                     if( i == 0 ) w = mDstFar;
                     if( i == x_size - 1 ) e = mDstFar;
                     if( j == 0 ) s = mDstFar;
                     if( j == y_size - 1 ) n = mDstFar;
                     phi( i, j ) = UpdatePoint2D( phi, 
                                                  i, j,
                                                  smallest,
                                                  smallest_direction,
                                                  e, w, s, n );
                     err += fabs( phi( i, j ) - c );*/
                  }
               }
            err /= x_size * y_size;
            if( verbose )
               cout << iter << " " << s1 << " " << 
                       s2 << " ERR: " << err << "    \r" << flush;
         }
   }
   if( ! _fixed ) delete fixed;
   return true;
}
