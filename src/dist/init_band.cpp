/***************************************************************************
                          init_band.cpp  -  description
                             -------------------
    begin                : 2005/08/10
    copyright            : (C) 2005 by Tomá¹ Oberhuber
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

#include <mdiff.h>
#include "init_band.h"
#include "debug.h"

//--------------------------------------------------------------------------
static void AddCubes( int i, int j,
                      const mGrid2D< double >& phi,
                      mVector< 2, int >* stack,
                      int& stack_end,
                      mField2D< char >& in_stack,
                      mField2D< bool >& visited_cubes )
{
   DBG_FUNCTION_NAME( "", "AddCubes" );
   int x_size = phi. GetXSize();
   int y_size = phi. GetYSize();
   
   if( visited_cubes( i, j ) ) return;
   visited_cubes( i, j ) = true;
   
   const double& h = phi. GetHx(); // we assume H(). x == H(). y
   char instck0( true ),
        instck1( true ),
        instck2( true ),
        instck3( true );
   
   instck0 = in_stack( i, j );
   if( i < x_size - 1 ) instck1 = in_stack( i + 1, j );
   if( i < x_size - 1 && j < y_size - 1 )
      instck2 = in_stack( i + 1, j + 1 );
   if( j < y_size - 1 ) instck3 = in_stack( i, j + 1 );
   if( instck0 && instck1 && instck2 && instck3 ) return;
   
   bool cross0( false ),
        cross1( false ),
        cross2( false ),
        cross3( false );
   
   if( i < x_size - 1 ) 
      cross0 = ( phi( i, j ) * phi( i + 1, j ) <= 0.0 );
   if( i < x_size - 1 && j < y_size - 1 )
   {
      cross1 = ( phi( i + 1, j ) * phi( i + 1, j + 1 ) <= 0.0 );
      cross2 = ( phi( i + 1, j + 1 ) * phi( i, j + 1 ) <= 0.0 );
   }
   if( j < y_size - 1 )
      cross3 = ( phi( i, j + 1 ) * phi( i, j ) <= 0.0 );
   
   if( ! instck0 && ( cross0 || cross3 ) )
   {
      int i_array[ 2 ] = { i, j };
      stack[ stack_end ++ ] = mVector< 2, int >( i_array );
      in_stack( i, j ) = 1;
      DBG_COUT( "Adding to stack: "  << i << " " << j );
   }
   
   if( ! instck1 && ( cross0 || cross1 ) )
   {
      int i_array[ 2 ] = { i + 1, j };
      stack[ stack_end ++ ] = mVector< 2, int >( i_array );
      in_stack( i + 1, j ) = 1;
      DBG_COUT( "Adding to stack: "  << i + 1 << " " << j );
   }
   
   if( ! instck2 && ( cross1 || cross2 ) )
   {
      int i_array[ 2 ] = { i + 1, j + 1 };
      stack[ stack_end ++ ] = mVector< 2, int >( i_array );
      in_stack( i + 1, j + 1 ) = 1;
      DBG_COUT( "Adding to stack: "  << i + 1 << " " << j + 1 );
   }
   
   if( ! instck3 && ( cross2 || cross3 ) )
   {
      int i_array[ 2 ] = { i, j + 1 };
      stack[ stack_end ++ ] = mVector< 2, int >( i_array );
      in_stack( i, j + 1 ) = 1;
      DBG_COUT( "Adding to stack: "  << i << " " << j + 1 );
   }

   /*if( cross0 || cross1 || cross2 || cross3 )
   {
      if( ! instck0 )
      {
         stack[ stack_end ++ ] = INDEX_2D( i, j );
         Val( in_stack, N_x, N_y, i, j ) = fixed;
#ifdef FMS_BAND_DBG
         band << i << " " << j << endl;
#endif
      }
      if( ! instck1 )
      {
         stack[ stack_end ++ ] = INDEX_2D( i + 1, j );
         Val( in_stack, N_x, N_y, i + 1, j ) = fixed;
#ifdef FMS_BAND_DBG
         band << i + 1 << " " << j << endl;
#endif
      }
      if( ! instck2 )
      {
         stack[ stack_end ++ ] = INDEX_2D( i + 1, j + 1 );
         Val( in_stack, N_x, N_y, i + 1, j + 1 ) = fixed;
#ifdef FMS_BAND_DBG
         band << i + 1 << " " << j + 1 << endl;
#endif
      }
      if( ! instck3 )
      {
         stack[ stack_end ++ ] = INDEX_2D( i, j + 1 );
         Val( in_stack, N_x, N_y, i, j + 1 ) = fixed;
#ifdef FMS_BAND_DBG
         band << i << " " << j + 1 << endl;
#endif
      }
   }*/
 
   if( i > 0 && cross3 ) // && ! Val( in_stack, N_x, N_y, i - 1, j ) )
      AddCubes( i - 1, j, phi, stack, stack_end, in_stack, visited_cubes );
   if( j > 0 && cross0 ) //&& ! Val( in_stack, N_x, N_y, i, j - 1 ) )
      AddCubes( i, j - 1, phi, stack, stack_end, in_stack, visited_cubes );
   if( i < x_size - 1 && cross1 ) //&& ! Val( in_stack, N_x, N_y, i + 1, j ) )
      AddCubes( i + 1, j, phi, stack, stack_end, in_stack, visited_cubes );
   if( j < y_size - 1 && cross2 ) // && ! Val( in_stack, N_x, N_y, i, j + 1 ) )
      AddCubes( i, j + 1, phi, stack, stack_end, in_stack, visited_cubes );
}
//--------------------------------------------------------------------------
static double GetRoot( const double& a, const double& b )
{
   assert( a * b <= 0.0 );
   return fabs( a / ( b - a ) );
}
//--------------------------------------------------------------------------
static double GetRoot2( const double& orig,
                        const double& a,
                        const double& b,
                        const double& h )
{
   assert( orig * a <= 0.0 && orig * b <= 0.0 );
   double a1 = h * GetRoot( orig, a );
   double b1 = h * GetRoot( orig, b );
   return fabs( a1 * b1 / sqrt( a1 * a1 + b1 * b1 ) );
}
//--------------------------------------------------------------------------
void RedistanceBand( mGrid2D< double >& phi,
                     mVector< 2, int >* stack,
                     int stack_end,
                     mField2D< char >& in_stack )
{
   DBG_FUNCTION_NAME( "", "RedistanceBand" );
   mGrid2D< double > tmp_phi( phi );
   int x_size = phi. GetXSize();
   int y_size = phi. GetYSize();
   const double h = phi. GetHx(); // we assume H. x == H. y
   int i, j;
   for( i = 0; i < x_size; i ++ )
      for( j = 0; j < y_size; j ++ )
         tmp_phi( i, j ) = 10.0 * h; 
         // no value in the stack can be bigger

   int k;
   for( k = 0; k < stack_end; k ++ )
   {
      bool e( false ), w( false ), s( false ), n( false );
      int i = stack[ k ][ 0 ];
      int j = stack[ k ][ 1 ];

      DBG_COUT( "i = " << i << " j = " << j
           << " f( i, j ) = " << phi( i, j ) );
      
      if( phi( i, j ) == 0.0 )
      {
         tmp_phi( i, j ) == 0.0;
         continue;
      }
      e = ( i < x_size - 1 && in_stack( i + 1, j ) &&
            phi( i, j ) * phi( i + 1, j ) <= 0.0 );
      w = ( i > 0 && in_stack( i - 1, j ) &&
            phi( i, j ) * phi( i - 1, j ) <= 0.0 );
      n = ( j < y_size - 1 && in_stack( i, j + 1 ) &&
            phi( i, j ) * phi( i, j + 1 ) <= 0.0 );
      s = ( j > 0 && in_stack( i, j - 1) &&
            phi( i, j ) * phi( i, j - 1 ) <= 0.0 );
      double root;
      assert( e || w || s || n );
      if( e ) 
      {
         root = h * GetRoot( phi( i, j ), phi( i + 1, j ) );
         if( fabs( root ) < fabs( tmp_phi( i, j ) ) )
         {
            DBG_COUT( "Smaller value comes from east: " << root );
            tmp_phi( i, j ) = fabs( root ) * Sign( phi( i, j ) );
         }
      }
      if( w ) 
      {
         root = h * GetRoot( phi( i, j ), phi( i - 1, j ) );
         if( fabs( root ) < fabs( tmp_phi( i, j ) ) )
         {
            DBG_COUT( "Smaller value comes from west: " << root );
            tmp_phi( i, j ) = fabs( root ) * Sign( phi( i, j ) );
         }
      }
      if( n ) 
      {
         root = h * GetRoot( phi( i, j ), phi( i, j + 1 ) );
         if( fabs( root ) < fabs( tmp_phi( i, j ) ) )
         {
            DBG_COUT( "Smaller value comes from north: " << root );
            tmp_phi( i, j ) = fabs( root ) * Sign( phi( i, j ) );
         }
      }
      if( s ) 
      {
         root = h * GetRoot( phi( i, j ), phi( i, j - 1 ) );
         if( fabs( root ) < fabs( tmp_phi( i, j ) ) )
         {
            DBG_COUT( "Smaller value comes from south: " << root );
            tmp_phi( i, j ) = fabs( root ) * Sign( phi( i, j ) );
         }
      }
      if( n && e )
      {
         root = GetRoot2( phi( i, j ), phi( i, j + 1 ), phi( i + 1, j ), h );
         if( fabs( root ) < fabs( tmp_phi( i, j ) ) )
         {
            DBG_COUT( "Smaller value comes from north-east: " << root );
            tmp_phi( i, j ) = fabs( root ) * Sign( phi( i, j ) );
         }
      }
      if( e && s )
      {
         root = GetRoot2( phi( i, j ), phi( i + 1, j ), phi( i, j - 1 ), h );
         if( fabs( root ) < fabs( tmp_phi( i, j ) ) )
         {
            DBG_COUT( "Smaller value comes from east-south: " << root );
            tmp_phi( i, j ) = fabs( root ) * Sign( phi( i, j ) );
         }
      }
      if( s && w )
      {
         root = GetRoot2( phi( i, j ), phi( i, j - 1 ), phi( i - 1, j ), h );
         if( fabs( root ) < fabs( tmp_phi( i, j ) ) )
         {
            DBG_COUT( "Smaller value comes from south-west: " << root );
            tmp_phi( i, j ) = fabs( root ) * Sign( phi( i, j ) );
         }
      }
      if( w && n )
      {
         root = GetRoot2( phi( i, j ), phi( i - 1, j ), phi( i, j + 1 ), h );
         if( fabs( root ) < fabs( tmp_phi( i, j ) ) )
         {
            DBG_COUT( "Smaller value comes from north-west: " << root );
            tmp_phi( i, j ) = fabs( root ) * Sign( phi( i, j ) );
         }
      }
   }
   for( k = 0; k < stack_end; k ++ )
   {
      int i = stack[ k ][ 0 ];
      int j = stack[ k ][ 1 ];
      phi( i, j ) = tmp_phi( i, j );
   }           
}
//--------------------------------------------------------------------------
void InitBand( const mGrid2D< double >& phi,
               mVector< 2, int >* stack,
               int& stack_end,
               mField2D< char >& in_stack )
{
   int x_size = phi. GetXSize();
   int y_size = phi. GetYSize();
   
   mField2D< bool > visited_cubes( x_size, y_size );
   visited_cubes. Zeros();

   int i, j;
   for( i = 0; i < x_size - 1; i ++ )
      for( j = 0; j < y_size - 1; j ++ )
      {
         if( phi( i, j ) * phi( i + 1, j ) <= 0.0 ||
             phi( i, j ) * phi( i, j + 1 ) <= 0.0 ||
             phi( i, j ) * phi( i + 1, j + 1 ) <= 0.0 )
               AddCubes( i, j, phi, stack, stack_end, in_stack, visited_cubes );
      }
}
//--------------------------------------------------------------------------
