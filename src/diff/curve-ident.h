/***************************************************************************
                          curve-ident.h  -  description
                             -------------------
    begin                : 2007/07/08
    copyright            : (C) 2007 by Tomá¹ Oberhuber
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

#ifndef curve_identH
#define curve_identH

#include <core/tnlCurve.h>
#include <diff/mGrid2D.h>
#include <debug/tnlDebug.h>

//! Supporting structure for a curve identification
// TODO replace it with tnlVector< 2, long int > 
struct MeshIndex
{
   long int i, j;
   MeshIndex( long int _i,
              long int _j )
      : i( _i ), j( _j ){}
};

template< class T > bool GetLevelSetCurve( const mGrid2D< T >& u,
                                           tnlCurve< tnlVector< 2, T > >& crv,
                                           const double level = 0.0 )
{
   dbgFunctionName( "", "GetLevelSetCurve" );
   long int i, j, k;
   const long int x_size = u. GetXSize();
   const long int y_size = u. GetYSize();

   // this list stores curves or just curve fargments
   tnlList< tnlList< MeshIndex>* > curves;

   // generating curves or fragments
   for( i = 0; i < x_size - 1; i ++ )
      for( j = 0; j < y_size - 1; j ++ )
      {
         const T a1 = u( i, j ) - level;
         const T a2 = u( i, j + 1 ) - level;
         const T a3 = u( i + 1, j + 1 ) - level;
         const T a4 = u( i + 1, j ) - level;
         if( a1 * a2 > 0.0 && a2 * a3 > 0.0 &&
             a3 * a4 > 0.0 && a4 * a1 > 0.0 )
            continue;
         // There is a curve going through the mesh.
         // Find if it is adjacent to begining or end of
         // some of already traced curves (or just curve fragment).
         dbgCout( "Curve detected at ( " << i << ", " << j << ")" );
         bool added( false );
         for( k = 0; k < curves. Size(); k ++ )
         {
            long int l = curves[ k ] -> Size();
            MeshIndex mi = ( * curves[ k ] )[ l - 1 ];
            long int n1 = abs( ( long int ) i - ( long int ) mi. i );
            long int n2 = abs( ( long int ) j - ( long int ) mi. j );
            if( ( n1 == 1 && n2 == 0 ) ||
                ( n1 == 0 && n2 == 1 ) )
            {
               curves[ k ] -> Append( MeshIndex( i, j ) );
               added = true;
               dbgCout( "Appending to list no. " << k << "; list size -> " <<
                         curves[ k ] -> Size() );
               break;
            }
            mi = ( * curves[ k ] )[ 0 ];
            n1 = abs( ( long int ) i - ( long int ) mi. i );
            n2 = abs( ( long int ) j - ( long int ) mi. j );
            if( ( n1 == 1 && n2 == 0 ) ||
                ( n1 == 0 && n2 == 1 ) )
            {
               curves[ k ] -> Prepend( MeshIndex( i, j ) );
               added = true;
               dbgCout( "Prepending to list no. " << k << "; list size ->  " << 
                     curves[ k ] -> Size() );
               break;
            }
         }
         // If it is not create new curve fragment.
         if( ! added )
         {
            tnlList< MeshIndex >* new_list = new tnlList< MeshIndex >;
            new_list -> Append( MeshIndex( i, j ) );
            curves. Append( new_list );
            dbgCout( "Adding new list." );
         }
      }
   
   // Now defragment all curves as much as it is possible.
   // It means - check if there are two curves whose begening and end
   // match, connect the together and erase the appended (or prepended) one.
   dbgCout( "Defragmenting lists ... ");
   bool fragmented( true );
   while( fragmented )
   {
      fragmented = false;
      for( i = 0; i < curves. Size(); i ++ )
      {
         tnlList< MeshIndex >& c1 = * curves[ i ];
         MeshIndex c1_start = c1[ 0 ];
         MeshIndex c1_end = c1[ c1. Size() - 1 ];
         for( j = 0 ; j < curves. Size(); j ++ )
         {
            if( i == j ) continue;
            tnlList< MeshIndex >& c2 = * curves[ j ];
            assert( &c2 != &c1 );
            MeshIndex c2_start = c2[ 0 ];
            MeshIndex c2_end = c2[ c2. Size() - 1 ];
            long int n1, n2;
            n1 = abs( ( long int ) c1_start. i - ( long int ) c2_end. i );
            n2 = abs( ( long int ) c1_start. j - ( long int ) c2_end. j );
            if( ( n1 == 1 && n2 == 0 ) ||
                ( n1 == 0 && n2 == 1 ) )
            {
               dbgCout( "Prepending the list no. " << j << 
                         " (" << c2. Size() <<") to the list no. " << i <<
                         " (" << c1. Size() <<").");
               c1. PrependList( c2 );
               curves. DeepErase( j );
               if( i > j ) i --;
               j --;
               c1_start = c2_start;
               dbgCout( "New list size is " << c1. Size() );
               fragmented = true;
               continue;
            }
            n1 = abs( ( long int ) c1_end. i - ( long int ) c2_start. i );
            n2 = abs( ( long int ) c1_end. j - ( long int ) c2_start. j );
            if( ( n1 == 1 && n2 == 0 ) ||
                ( n1 == 0 && n2 == 1 ) )
            {
               dbgCout( "Appending the list no. " << j <<
                         " (" << c2. Size() <<") to the list no. " << i <<
                         " (" << c1. Size() <<").");
               c1. AppendList( c2 );
               curves. DeepErase( j );
               if( i > j ) i --;
               j --;
               c1_end = c2_end;
               dbgCout( "New list size is " << c1. Size() );
               fragmented = true;
               continue;
            }
            n1 = abs( ( long int ) c1_start. i - ( long int ) c2_start. i );
            n2 = abs( ( long int ) c1_start. j - ( long int ) c2_start. j );
            if( ( n1 == 1 && n2 == 0 ) ||
                ( n1 == 0 && n2 == 1 ) )
            {
               dbgCout( "Prepending (reversaly) the list no. " << j <<
                         " (" << c2. Size() <<" ) to the list no. " << i <<
                         " (" << c1. Size() <<" ).");
               for( k = 0; k < c2. Size(); k ++ )
                  c1. Prepend( c2[ k ] );
               curves. DeepErase( j );
               if( i > j ) i --;
               j --;
               c1_start = c2_end;
               dbgCout( "New list size is " << c1. Size() );
               fragmented = true;
               continue;
            }
            n1 = abs( ( long int ) c1_end. i - ( long int ) c2_end. i );
            n2 = abs( ( long int ) c1_end. j - ( long int ) c2_end. j );
            if( ( n1 == 1 && n2 == 0 ) ||
                ( n1 == 0 && n2 == 1 ) )
            {
               dbgCout( "Appending (reversaly) the list no. " << j <<
                         " (" << c2. Size() <<" ) to the list no. " << i <<
                         " (" << c1. Size() <<" ).");
               for( k = c2.Size(); k > 0; k -- )
                  c1. Append( c2[ k - 1 ] );
               curves. DeepErase( j );
               if( i > j ) i --;
               j --;
               c1_end = c2_start;
               dbgCout( "New list size is " << c1. Size() );
               fragmented = true;
               continue;
            }
         }
      }
   }
   dbgCout( "There are " << curves. Size() << " curves now." );
   // Now check if the curves are closed ( the begining and
   // the end match).
   for( i = 0; i < curves. Size(); i ++ )
   {
      tnlList< MeshIndex >& c = * curves[ i ];
      long int l = c. Size();
      MeshIndex m1 = c[ 0 ];
      MeshIndex m2 = c[ l - 1 ];
      long int n1 = abs( ( long int ) m1. i - ( long int ) m2. i );
      long int n2 = abs( ( long int ) m1. j - ( long int ) m2. j );
      if( ( n1 == 1 && n2 == 0 ) ||
          ( n1 == 0 && n2 == 1 ) )
      {
         dbgCout( "Closing curve no. " << i );
         c. Append( m1 );
      }
   }
 
   // Count precisly the curves points
   const double a_x = u. GetAx();
   const double a_y = u. GetAy();
   const double h_x = u. GetHx();
   const double h_y = u. GetHy();
   tnlVector< 2, T > null_vector;
   if( ! crv. IsEmpty() )
      crv. Append( null_vector, true ); //separator
   for( i = 0; i < curves. Size(); i ++ )
   {
      if( i > 0 ) crv. Append( null_vector, true );  //separator
      tnlList< MeshIndex >& c = * curves[ i ];
      long int l = c. Size();
      tnlVector< 2, T > first;
      for( j = 0; j < l - 1; j ++ )
      {
         MeshIndex m1 = c[ j ];
         MeshIndex m2 = c[ j + 1 ];
         long int n1 = m2. i - m1. i;
         long int n2 = m2. j - m1. j;
         T p[ 2 ], v[ 2 ];
         if( n1 == 0 && n2 == -1 )
         {
            p[ 0 ] = a_x + m1. i * h_x;
            p[ 1 ] = a_y + m1. j * h_y;
            v[ 0 ] = u( m1. i, m1. j ) - level;
            v[ 1 ] = u( m1. i + 1, m1. j ) - level;
         }
         if( n1 == 1 && n2 == 0 )
         {
            p[ 0 ] = a_x + ( m1. i + 1 ) * h_x;
            p[ 1 ] = a_y + ( m1. j ) * h_y;
            v[ 0 ] = u( m1. i + 1, m1. j ) - level;
            v[ 1 ] = u( m1. i + 1, m1. j + 1 ) - level;
         }
         if( n1 == 0 && n2 == 1 )
         {
            p[ 0 ] = a_x + ( m1. i + 1 ) * h_x;
            p[ 1 ] = a_y + ( m1. j + 1 ) * h_y;
            v[ 0 ] = u( m1. i + 1, m1. j + 1 ) - level;
            v[ 1 ] = u( m1. i, m1. j + 1 ) - level;
         }
         if( n1 == -1 && n2 == 0 )
         {
            p[ 0 ] = a_x + ( m1. i ) * h_x;
            p[ 1 ] = a_y + ( m1. j + 1 ) * h_y;
            v[ 0 ] = u( m1. i, m1. j + 1 ) - level;
            v[ 1 ] = u( m1. i, m1. j ) - level;
         }
         if( v[ 0 ] * v[ 1 ] > 0.0 )
         {
            // this can happen near the edge or when two curves cross each other
            //curve. Append( CurveElement( Vector2D( 0.0, 0.0 ), true ) );
            //cerr << "Warning: v1 * v2 > 0 in curve identification." << endl;
            continue;
         }
         T r = v[ 0 ] / ( v[ 1 ] - v[ 0 ] );
         p[ 0 ] += ( ( T ) n2  ) * r * h_x;
         p[ 1 ] += ( ( T ) -n1  ) * r * h_y;
         crv. Append( tnlVector< 2, T >( p ) );
         if( j == 0 ) first = tnlVector< 2, T >( p );
      }
      MeshIndex m1 = c[ 0 ];
      MeshIndex m2 = c[ l - 1 ];
      if( m1. i == m2. i && m1. j == m2. j  )
         crv. Append( first );
   }

   curves. DeepEraseAll();
   return true;
};
                                           

#endif
