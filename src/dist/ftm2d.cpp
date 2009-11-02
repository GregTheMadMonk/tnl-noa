/***************************************************************************
                          ftm2d.cpp  -  description
                             -------------------
    begin                : 2005/08/12
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

#include <float.h>
#include <mdiff.h>
#include "debug.h"
#include "direct.h"
#include "ftm2d.h"
#include "init_band.h"

//--------------------------------------------------------------------------
void DrawStack( const mGrid2D< double >& phi, 
                mVector< 2, int > * stack,
                int stack_end,
                const char* file_name )
{
   fstream file;
   file. open( file_name, ios :: out );
   if( ! file )
   {
      cerr << "Unable to open file " << file_name << endl;
      return;
   }
   int i;
   const double h = phi. GetHx();
   const double a_x = phi. GetAx();
   const double a_y = phi. GetAy();
   for( i = 0; i < stack_end; i ++ )
   {
      file << a_x + stack[ i ][ 0 ] * h << " " <<
              a_y + stack[ i ][ 1 ] * h << " " << 
              phi( stack[ i ][ 0 ], stack[ i ][ 1 ] ) << endl;
      
   }
   file. close();
}
//--------------------------------------------------------------------------
double UpdatePoint( mGrid2D< double >& phi, 
                    int i, int j,
                    mField2D< char >& in_stack )
{
   DBG_FUNCTION_NAME( "", "UpdateNeighbour" );
   // Here we have a position of a point to update so
   // we look for its neigbbours to figure out which
   // can be used and we look for the one with the smallest
   // value (and its direction relatively to the given point)
   int x_size = phi. GetXSize();
   int y_size = phi. GetYSize();
   mDstState e( mDstFar ), w( mDstFar ), s( mDstFar ), n( mDstFar );
   if( i < x_size - 1 && in_stack( i + 1, j ) != 0 )
      e = mDstTentative;
   if( i > 0 && in_stack( i - 1, j ) != 0 )
      w = mDstTentative;
   if( j > 0 && in_stack( i, j - 1 ) != 0 )
      s = mDstTentative;
   if( j < y_size - 1 && in_stack( i, j + 1 ) != 0 )
      n = mDstTentative;
   DBG_COND_EXPR( e == mDstTentative, e );
   DBG_COND_EXPR( e == mDstTentative, phi( i + 1, j ) );
   DBG_COND_EXPR( e == mDstTentative, in_stack( i + 1, j ) );
   DBG_COND_EXPR( w == mDstTentative, w );
   DBG_COND_EXPR( w == mDstTentative, phi( i - 1, j ) );
   DBG_COND_EXPR( w == mDstTentative, in_stack( i - 1, j ) );
   DBG_COND_EXPR( s == mDstTentative, s );
   DBG_COND_EXPR( s == mDstTentative, phi( i, j - 1 ) );
   DBG_COND_EXPR( s == mDstTentative, in_stack( i, j - 1 ) );
   DBG_COND_EXPR( n == mDstTentative, n );
   DBG_COND_EXPR( n == mDstTentative, phi( i, j + 1 ) );
   DBG_COND_EXPR( n == mDstTentative, in_stack( i, j + 1 ) );

   double smallest( DBL_MAX );
   mDstDirection smallest_direction;
   if( e == mDstTentative && 
       fabs( phi( i + 1, j ) ) < fabs( smallest ) )
   {
      smallest = phi( i + 1, j );
      smallest_direction = mDstEast;
   }
   if( w == mDstTentative && 
       fabs( phi( i - 1, j ) ) < fabs( smallest ) )
   {
      smallest = phi( i - 1, j );
      smallest_direction = mDstWest;
   }
   if( n == mDstTentative && 
       fabs( phi( i, j + 1 ) ) < fabs( smallest ) )
   {
      smallest = phi( i, j + 1 );
      smallest_direction = mDstNorth;
   }
   if( s == mDstTentative && 
       fabs( phi( i, j - 1 ) ) < fabs( smallest ) )
   {
      smallest = phi( i, j - 1 );
      smallest_direction = mDstSouth;
   }
   DBG_EXPR( smallest );
   DBG_EXPR( smallest_direction );
   DBG_EXPR( phi( i, j ) );

   return UpdatePoint2D( phi,
                         i, j,
                         smallest,
                         smallest_direction,
                         e, w, s, n );
}
//--------------------------------------------------------------------------
void UpdateNeighbour( mGrid2D< double >& phi,
                      mVector< 2, int >* stack,
                      int& cur_stack_end,
                      mField2D< char >& in_stack,
                      int i, int j,
                      mDstDirection dir,
                      double& u_max,
                      bool correct,
                      char pass )
{
   DBG_FUNCTION_NAME( "", "UpdateNeighbour" );
   int x_size = phi. GetXSize();
   int y_size = phi. GetYSize();
   double v;
  
   DBG_EXPR( i );
   DBG_EXPR( j );
   DBG_EXPR( ( int ) pass );

   // Here we have a point and direction of a neighbour that
   // we may update. We check whether this point is inside the grid
   // (if it exists) and if it is not fixed already.
   if( dir == mDstWest && 
       i > 0 && 
       in_stack( i - 1, j ) < pass )
   {
      DBG_EXPR( phi( i - 1, j ) );
      if( correct == 0 )
      {
         phi( i - 1, j ) = UpdatePoint( phi, i - 1, j, in_stack );
         u_max = Max( u_max, fabs( phi( i - 1, j ) ) );
         if( in_stack( i - 1, j ) == 0 )
         {
            int i_array[ 2 ] = { i - 1, j };
            stack[ cur_stack_end ++ ] = mVector< 2, int >( i_array );
         }
         in_stack( i - 1, j ) = pass;
      }
      else
      {
         v = UpdatePoint( phi, i - 1, j, in_stack );
         if( fabs( v ) < u_max )
         {
            phi( i - 1, j ) = v;
            if(  in_stack( i - 1, j ) == 0 )
            {
               int i_array[ 2 ] = { i - 1, j };
               stack[ cur_stack_end ++ ] = mVector< 2, int >( i_array );
            }
            in_stack( i - 1, j ) = pass;
         }
      }
   }
   if( dir == mDstSouth &&
       j > 0 && 
       in_stack( i, j - 1 ) < pass )
   {
      DBG_EXPR( phi( i, j -1 ) );
      if( correct == 0 )
      {
         phi( i, j - 1 ) = UpdatePoint( phi, i, j - 1, in_stack );
         u_max = Max( u_max, fabs( phi( i, j - 1 ) ) );
         if( in_stack( i, j - 1 ) == 0 )
         {
            int i_array[ 2 ] = { i, j - 1 };
            stack[ cur_stack_end ++ ] = mVector< 2, int >( i_array );
         }
         in_stack( i, j - 1 ) = pass;
      }
      else
      {
         v = UpdatePoint( phi, i, j - 1, in_stack );
         if( fabs( v ) < u_max )
         {
            phi( i, j - 1 ) = v;
            if( in_stack( i, j - 1 ) == 0 )
            {
               int i_array[ 2 ] = { i, j - 1 };
               stack[ cur_stack_end ++ ] = mVector< 2, int >( i_array );
            }
            in_stack( i, j - 1 ) = pass;
         }
      }
   }
   if( dir == mDstEast && 
       i < x_size - 1 && 
       in_stack( i + 1, j ) < pass )
   {
      DBG_EXPR( phi( i + 1, j ) );
      if( correct == 0 )
      {
         phi( i + 1, j ) = UpdatePoint( phi, i + 1, j, in_stack );
         u_max = max( u_max, fabs( phi( i + 1, j ) ) );
         if( in_stack( i + 1, j ) == 0 )
         {
            int i_array[ 2 ] = { i + 1, j };
            stack[ cur_stack_end ++ ] = mVector< 2, int >( i_array );
         }
         in_stack( i + 1, j ) = pass;
      }
      else
      {
         v = UpdatePoint( phi, i + 1, j, in_stack );
         if( fabs( v ) < u_max )
         {
            phi( i + 1, j ) = v;
            if( in_stack( i + 1, j ) == 0 )
            {
               int i_array[ 2 ] = { i + 1, j };
               stack[ cur_stack_end ++ ] = mVector< 2, int >( i_array );
            }
            in_stack( i + 1, j ) = pass;
         }
      }
   }
   if( dir == mDstNorth && 
       j < y_size - 1 &&
       in_stack( i, j + 1 ) < pass )
   {
      DBG_EXPR( phi( i, j + 1 ) );
      if( correct == 0 )
      {
         phi( i, j + 1 ) = UpdatePoint( phi, i, j + 1, in_stack );
         u_max = Max( u_max, fabs( phi( i, j + 1 ) ) );
         if( in_stack( i, j + 1 ) == 0 )
         {
            int i_array[ 2 ] = { i, j + 1 };
            stack[ cur_stack_end ++ ] = mVector< 2, int >( i_array );
         }
         in_stack( i, j + 1 ) = pass;
      }
      else
      {
         v = UpdatePoint( phi, i, j + 1, in_stack );
         if( fabs( v ) < u_max )
         {
            phi( i, j + 1 ) = v;
            if( in_stack( i, j + 1 ) == 0 )
            {
               int i_array[ 2 ] = { i, j + 1 };
               stack[ cur_stack_end ++ ] = mVector< 2, int >( i_array );
            }
            in_stack( i, j + 1 ) = pass;
         }
      }
   }
}
//--------------------------------------------------------------------------
/*static int Dir2Pos( int x_size,
                    int y_size,
                    int i, int j,
                    DIRECTION dir, int& new_i, int& new_j  )
{
   switch( dir )
   {
      case mDstEast:
         if( i < x_size - 1 )
         {
            new_i = i + 1,
            new_j = j;
            return 1;
         }
         return 0;
      case mDstWest:
         if( i > 0 )
         {
            new_i = i - 1;
            new_j = j;
            return 1;
         }
         return 0;
      case mDstNorth:
         if( j < y_size - 1 )
         {
            new_i = i;
            new_j = j + 1;
            return 1;
         }
         return 0;
      case mDstSouth:
         if( j > 0 )
         {
            new_i = i;
            new_j = j - 1;
            return 1;
         }
         return 0;
   }
   return 0;
}*/
//--------------------------------------------------------------------------
static void UpdateNeighbours( mGrid2D< double >& phi,
                              mVector< 2, int >* stack,
                              int& cur_stack_end,
                              mField2D< char >& in_stack,
                              int i, int j, 
                              double& u_max,
                              bool correct,
                              char pass )
{
   int x_size = phi. GetXSize();
   int y_size = phi. GetYSize();

   // this function just try to update neighbours
   // we do not check which is going to be really 
   // updated yet

   UpdateNeighbour( phi, stack,
                    cur_stack_end, in_stack,
                    i, j, mDstWest, u_max,
                    correct, pass );
   UpdateNeighbour( phi, stack,
                    cur_stack_end, in_stack,
                    i, j, mDstSouth, u_max,
                    correct, pass );
   UpdateNeighbour( phi, stack,
                    cur_stack_end, in_stack,
                    i, j, mDstEast, u_max,
                    correct, pass );
   UpdateNeighbour( phi, stack,
                    cur_stack_end, in_stack,
                    i, j, mDstNorth, u_max,
                    correct, pass );

}
//--------------------------------------------------------------------------
static void FrontTracing( mGrid2D< double >& phi,
                          mVector< 2, int >* stack,
                          int cur_stack_end,
                          mField2D< char >& in_stack,
                          const double& band_width )
{
   DBG_FUNCTION_NAME( "", "FrontTracing" );
   int x_size = phi. GetXSize();
   int y_size = phi. GetYSize();
   
   int stack_beg( 0 );
   int init_stack = cur_stack_end;
   int stack_end = cur_stack_end;

   // compute until we update all the points of the grid
   while( cur_stack_end < x_size * y_size )
   {
      //if( verbose )
         cout << "Stack beg. " << stack_beg << " stack end " <<
                 stack_end << "/ " << x_size * y_size << "    \r" <<
                 flush;
      double u_max( 0.0 );
      // we perform two passes each going in other direction
      for( int k = stack_beg; k < stack_end; k ++ )
      {
         int i = stack[ k ][ 0 ];
         int j = stack[ k ][ 1 ];
         assert( i < x_size && j < y_size );
         // try to update all neighbours
         UpdateNeighbours( phi, stack, cur_stack_end,
                           in_stack, i, j, u_max, 0, 1 );
      }
      for( int k = stack_end - 1; k > stack_beg; k -- )
      {
         int i = stack[ k ][ 0 ];
         int j = stack[ k ][ 1 ];
         assert( i < x_size && j < y_size );
         // try to update all neighbours
         UpdateNeighbours( phi, stack, cur_stack_end,
                           in_stack, i, j, u_max, 0, 2 );
      }
      stack_beg = stack_end;
      stack_end = cur_stack_end;
      //DBG_CALL( DrawStack( phi, stack, stack_end, "stack" ) );
      //DBG_WAIT();

      // synchronizing step
      // it is necesary to synchronize the virtual front with
      // the real one - we take maximum attained during the first
      // to passes and proceed only at those parts of the front
      // where the actual value is lower then alredy attained maximum,
      // so it will avoids large differnces of values along the front
      cout << "Stack beg. " << stack_beg << " stack end " << 
              stack_end << "/ " << x_size * y_size << "    \r" << 
              flush;

      int stack_beg_tmp( stack_beg );
      while( stack_beg_tmp < stack_end )
      {
         for( int k = stack_beg_tmp; k < stack_end; k ++ )
         {
            int i = stack[ k ][ 0 ];
            int j = stack[ k ][ 1 ];
            assert( i < x_size && j < y_size );
            UpdateNeighbours( phi, stack, cur_stack_end,
                              in_stack, i, j, u_max, 1, 1 );

         }
         for( int k = stack_end - 1; k > stack_beg_tmp; k -- )
         {
            int i = stack[ k ][ 0 ];
            int j = stack[ k ][ 1 ];
            assert( i < x_size && j < y_size );
            UpdateNeighbours( phi, stack, cur_stack_end,
                              in_stack, i, j, u_max, 1, 2 );

         }
         stack_beg_tmp = stack_end;
         stack_end = cur_stack_end;
      }

      //DBG_CALL( DrawStack( phi, stack, stack_end, "stack" ) );
      //DBG_WAIT();
      
      if( band_width && u_max > band_width )
         return;
      
   }
}

//--------------------------------------------------------------------------
void DstFrontTracing2D( mGrid2D< double >& phi, 
                        const double& band_width )
{
   DBG_FUNCTION_NAME( "", "DstFrontTracing2D" );
   // this function finds initial narrow band ( just the points
   // of the grids having a neighbour with oposite sign compute 
   // distance function in the grid and then starts front tracing
   // method
   int x_size = phi. GetXSize();
   int y_size = phi. GetYSize();
   
   mVector< 2, int >* stack = new mVector< 2, int >[ x_size * y_size ];
   int stack_beg( 0 ), stack_end( 0 );
   mField2D< char > in_stack( x_size, y_size );
   in_stack. Zeros();

 
   // finding initial band
   InitBand( phi, stack, stack_end, in_stack );  

   // compute the distance function
   RedistanceBand( phi, stack, stack_end, in_stack );
   
   // set the flags inside the band to a large number
   // so it stay fixed during the computation
   int i;
   for( i = 0; i < stack_end; i ++ )
      in_stack( stack[ i ][ 0 ], stack[ i ][ 1 ] ) = 64;
   
   //DBG_CALL( DrawStack( phi, stack, stack_end, "stack" ) );
   //DBG_WAIT();
   
   // run front tracing
   FrontTracing( phi, stack, stack_end, in_stack, band_width );
   delete[] stack;
}
