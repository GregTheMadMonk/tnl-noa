/***************************************************************************
                          fmm2d.cpp  -  description
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

#include <debug/tnlDebug.h>
#include <diff/mGrid2D.h>
#include <core/mList.h>
#include <core/mVector.h>
#include <debug/tnlDebug.h>
#include "direct.h"
#include "fmm2d.h"
#include "init_band.h"

//--------------------------------------------------------------------------
static void Recompute( mGrid2D< double >& phi,
                       int i,
                       int j,
                       const double& smallest,
                       mDstDirection smallest_direction,
                       mList< mVector< 2, int > >& band,
                       mField2D< mDstState >& state_field )
{
   dbgFunctionName( "", "Recompute" );
   if( state_field( i, j ) == mDstFixed ) return;
   dbgCout( "Recomputing point: " << i << " " << j << " -> " << phi( i, j ) );
   
   double sgn;
   if( state_field( i, j ) == mDstFar )
   {
      state_field( i, j ) = mDstTentative;
      int i_array[ 2 ];
      i_array[ 0 ] = i;
      i_array[ 1 ] = j;
      band. Append( mVector< 2, int >( i_array ) );
      sgn = Sign( smallest );
   }
   else sgn = Sign( phi( i, j ) );
   assert( phi. GetHx() == phi. GetHy() );
   const double& h = phi. GetHx(); 
   mDstState e( mDstFar ),
             w( mDstFar ),
             n( mDstFar ),
             s( mDstFar );
   if( i < phi. GetXSize() - 1 ) e = state_field( i + 1, j );
   if( i > 0 ) w = state_field( i - 1, j );
   if( j < phi. GetYSize() - 1 ) n = state_field( i, j + 1 );
   if( j > 0 ) s = state_field( i, j - 1 );

   dbgCout( " e = " << e << " w = " << w <<
             " s = " << s << " n = " << n );
   assert( e != mDstFar || w != mDstFar || n != mDstFar || s != mDstFar );
   
   phi( i, j ) = UpdatePoint2D( phi,
                                i, j,
                                smallest,
                                smallest_direction,
                                e, w, s, n );

   dbgExpr( phi( i, j ) );
   return;
}
//--------------------------------------------------------------------------
void DstFastMarching2D( mGrid2D< double >& phi, 
                        const double& band_width,
                        const double& delta )
{
   dbgFunctionName( "", "FastMarching2D" );
   dbgExpr( delta );
   dbgExpr( band_width );
 
   mList< mVector< 2, int > > band;
   
   int x_size = phi. GetXSize();
   int y_size = phi. GetYSize();
 
   // Init stack redistancing
   int stack_end( 0 );
   mField2D< char > in_stack( x_size, y_size );
   mVector< 2, int >* stack = new mVector< 2, int >[ x_size * y_size ];
   in_stack. Zeros();

   InitBand( phi, stack, stack_end, in_stack );  
   RedistanceBand( phi, stack, stack_end, in_stack );
   
   mField2D< mDstState > state_field( x_size, y_size );
   
   int i, j, fixed( stack_end );
   for( i = 0; i < x_size; i ++ )
      for( j = 0; j < y_size; j ++ )
         if( in_stack( i, j ) )
         {
            state_field( i, j ) = mDstFixed;
            int i_array[ 2 ];
            i_array[ 0 ] = i;
            i_array[ 1 ] = j;
            band. Append( mVector< 2, int >( i_array ) );
         }
         else state_field( i, j ) = mDstFar;

   if( band. IsEmpty() )
      cerr << "Initial band is empty!" << endl;

   if( band_width )
      cout << "Band width is set to " << band_width << endl;
   
   while( ! band. IsEmpty() )
   {
      int i;
      mVector< 2, int > smallest = band[ 0 ];
      int smallest_pos( 0 );
      for( i = 0; i < band. Size(); i ++ )
      {
         //dbgExpr( band[ i ] );
         double val = phi( band[ i ][ 0 ], band[ i ][ 1 ] );
         dbgExpr( val );
         
         if( fabs( val ) < fabs( phi( smallest[ 0 ], smallest[ 1 ] ) ) )
         {
            int _i = band[ i ][ 0 ];
            int _j = band[ i ][ 1 ];
            if( ( _i > 0 && 
                 ( state_field( _i - 1, _j ) == mDstFixed ||
                   phi( _i, _j ) * phi( _i - 1, _j ) < 0.0 ) ) ||
                ( _i < x_size - 1 && 
                  ( state_field( _i + 1, _j ) == mDstFixed || 
                    phi( _i, _j ) * phi( _i + 1, _j ) < 0.0 ) ) ||
                ( _j > 0 && 
                  ( state_field( _i, _j - 1 ) == mDstFixed ||
                    phi( _i, _j ) * phi( _i, _j - 1 ) < 0.0 ) ) ||
                ( _j < y_size - 1 && 
                  ( state_field( _i, _j + 1 ) == mDstFixed ||
                    phi( _i, _j ) * phi( _i, _j + 1 ) < 0.0 ) ) )
            {
               smallest = band[ i ];
               smallest_pos = i;
            }
         }
      }
      //dbgExpr( smallest );
      dbgExpr( phi( smallest[ 0 ], smallest[ 1 ] ) );
      
      //Add( alive_points, smallest -> data );
      int s_i = smallest[ 0 ];
      int s_j = smallest[ 1 ];
      band. Erase( smallest_pos );
      state_field( s_i, s_j ) = mDstFixed;
      fixed ++;
      
      const double& sm_val = phi( s_i, s_j );
      if( s_i < x_size - 1 && state_field( s_i + 1, s_j ) != mDstFixed )
         Recompute( phi, s_i + 1, s_j, sm_val, mDstWest, band, state_field );
      if( s_i > 0 && state_field( s_i - 1, s_j ) != mDstFixed )
         Recompute( phi, s_i - 1, s_j, sm_val, mDstEast, band, state_field );
      if( s_j < y_size - 1 && state_field( s_i, s_j + 1 ) != mDstFixed )
         Recompute( phi, s_i, s_j + 1, sm_val, mDstSouth, band, state_field );
      if( s_j > 0 && state_field( s_i, s_j - 1 ) != mDstFixed )
         Recompute( phi, s_i, s_j - 1, sm_val, mDstNorth, band, state_field );

      if( band_width )
         cout << fabs( phi( s_i, s_j ) ) << " / " << band_width << " ";
      cout << fixed << " / " << x_size * y_size << " - " << ( int )
         ( ( double ) fixed / ( double ) ( x_size * y_size ) * 100 )
         << " % done.\r" << flush;

      if( band_width && ( band_width < fabs( phi( s_i, s_j ) ) ) )
      {
         cout << endl << fabs( phi( s_i, s_j ) ) << endl;
         return;
      }
   }
   delete[] stack;
}
