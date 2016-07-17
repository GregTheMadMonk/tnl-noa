/***************************************************************************
                          curve-ident.h  -  description
                             -------------------
    begin                : 2007/07/08
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef curve_identH
#define curve_identH

#include <core/tnlCurve.h>
#include <legacy/mesh/tnlGridOld.h>
#include <debug/tnlDebug.h>

template< typename Real, typename Device, typename Index >
bool getLevelSetCurve( const tnlGridOld< 2, Real, Device, Index >& u,
                       tnlCurve< tnlStaticVector< 2, Real > >& crv,
                       const Real level = 0.0 )
{
   dbgFunctionName( "", "GetLevelSetCurve" );

   const Index xSize = u. getDimensions(). x();
   const Index ySize = u. getDimensions(). y();

   // this list stores curves or just curve fargments
   tnlList< tnlList< tnlStaticVector< 2, Index > >* > curves;

   // generating curves or fragments
   for( Index i = 0; i < xSize - 1; i ++ )
      for( Index j = 0; j < ySize - 1; j ++ )
      {
         const Real a1 = u. getElement( i, j ) - level;
         const Real a2 = u. getElement( i, j + 1 ) - level;
         const Real a3 = u. getElement( i + 1, j + 1 ) - level;
         const Real a4 = u. getElement( i + 1, j ) - level;
         if( a1 * a2 > Real( 0.0 ) && a2 * a3 > Real( 0.0 ) &&
             a3 * a4 > Real( 0.0 ) && a4 * a1 > Real( 0.0 ) )
            continue;
         // There is a curve going through the mesh.
         // Find if it is adjacent to begining or end of
         // some of already traced curves (or just curve fragment).
         dbgCout( "Curve detected at ( " << i << ", " << j << ")" );
         bool added( false );
         for( Index k = 0; k < curves. getSize(); k ++ )
         {
            Index l = curves[ k ] -> getSize();
            tnlStaticVector< 2, Index > mi = ( * curves[ k ] )[ l - 1 ];
            Index n1 = abs( ( Index ) i - ( Index ) mi. x() );
            Index n2 = abs( ( Index ) j - ( Index ) mi. y() );
            if( ( n1 == 1 && n2 == 0 ) ||
                ( n1 == 0 && n2 == 1 ) )
            {
               curves[ k ] -> Append( tnlStaticVector< 2, Index >( i, j ) );
               added = true;
               dbgCout( "Appending to list no. " << k << "; list size -> " <<
                         curves[ k ] -> getSize() );
               break;
            }
            mi = ( * curves[ k ] )[ 0 ];
            n1 = abs( ( Index ) i - ( Index ) mi. x() );
            n2 = abs( ( Index ) j - ( Index ) mi. y() );
            if( ( n1 == 1 && n2 == 0 ) ||
                ( n1 == 0 && n2 == 1 ) )
            {
               curves[ k ] -> Prepend( tnlStaticVector< 2, Index >( i, j ) );
               added = true;
               dbgCout( "Prepending to list no. " << k << "; list size ->  " <<
                     curves[ k ] -> getSize() );
               break;
            }
         }
         // If it is not create new curve fragment.
         if( ! added )
         {
            tnlList< tnlStaticVector< 2, Index > >* new_list = new tnlList< tnlStaticVector< 2, Index > >;
            new_list -> Append( tnlStaticVector< 2, Index >( i, j ) );
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
      for( Index i = 0; i < curves. getSize(); i ++ )
      {
         tnlList< tnlStaticVector< 2, Index > >& c1 = * curves[ i ];
         tnlStaticVector< 2, Index > c1_start = c1[ 0 ];
         tnlStaticVector< 2, Index > c1_end = c1[ c1. getSize() - 1 ];
         for( Index j = 0 ; j < curves. getSize(); j ++ )
         {
            if( i == j ) continue;
            tnlList< tnlStaticVector< 2, Index > >& c2 = * curves[ j ];
            assert( &c2 != &c1 );
            tnlStaticVector< 2, Index > c2_start = c2[ 0 ];
            tnlStaticVector< 2, Index > c2_end = c2[ c2. getSize() - 1 ];
            Index n1, n2;
            n1 = abs( ( Index ) c1_start. x() - ( Index ) c2_end. x() );
            n2 = abs( ( Index ) c1_start. y() - ( Index ) c2_end. y() );
            if( ( n1 == 1 && n2 == 0 ) ||
                ( n1 == 0 && n2 == 1 ) )
            {
               dbgCout( "Prepending the list no. " << j <<
                         " (" << c2. getSize() <<") to the list no. " << i <<
                         " (" << c1. getSize() <<").");
               c1. PrependList( c2 );
               curves. DeepErase( j );
               if( i > j ) i --;
               j --;
               c1_start = c2_start;
               dbgCout( "New list size is " << c1. getSize() );
               fragmented = true;
               continue;
            }
            n1 = abs( ( Index ) c1_end. x() - ( Index ) c2_start. x() );
            n2 = abs( ( Index ) c1_end. y() - ( Index ) c2_start. y() );
            if( ( n1 == 1 && n2 == 0 ) ||
                ( n1 == 0 && n2 == 1 ) )
            {
               dbgCout( "Appending the list no. " << j <<
                         " (" << c2. getSize() <<") to the list no. " << i <<
                         " (" << c1. getSize() <<").");
               c1. AppendList( c2 );
               curves. DeepErase( j );
               if( i > j ) i --;
               j --;
               c1_end = c2_end;
               dbgCout( "New list size is " << c1. getSize() );
               fragmented = true;
               continue;
            }
            n1 = abs( ( Index ) c1_start. x() - ( Index ) c2_start. x() );
            n2 = abs( ( Index ) c1_start. y() - ( Index ) c2_start. y() );
            if( ( n1 == 1 && n2 == 0 ) ||
                ( n1 == 0 && n2 == 1 ) )
            {
               dbgCout( "Prepending (reversaly) the list no. " << j <<
                         " (" << c2. getSize() <<" ) to the list no. " << i <<
                         " (" << c1. getSize() <<" ).");
               for( Index k = 0; k < c2. getSize(); k ++ )
                  c1. Prepend( c2[ k ] );
               curves. DeepErase( j );
               if( i > j ) i --;
               j --;
               c1_start = c2_end;
               dbgCout( "New list size is " << c1. getSize() );
               fragmented = true;
               continue;
            }
            n1 = abs( ( Index ) c1_end. x() - ( Index ) c2_end. x() );
            n2 = abs( ( Index ) c1_end. y() - ( Index ) c2_end. y() );
            if( ( n1 == 1 && n2 == 0 ) ||
                ( n1 == 0 && n2 == 1 ) )
            {
               dbgCout( "Appending (reversaly) the list no. " << j <<
                         " (" << c2. getSize() <<" ) to the list no. " << i <<
                         " (" << c1. getSize() <<" ).");
               for( Index k = c2.getSize(); k > 0; k -- )
                  c1. Append( c2[ k - 1 ] );
               curves. DeepErase( j );
               if( i > j ) i --;
               j --;
               c1_end = c2_start;
               dbgCout( "New list size is " << c1. getSize() );
               fragmented = true;
               continue;
            }
         }
      }
   }
   dbgCout( "There are " << curves. getSize() << " curves now." );
   // Now check if the curves are closed ( the begining and
   // the end match).
   for( Index i = 0; i < curves. getSize(); i ++ )
   {
      tnlList< tnlStaticVector< 2, Index > >& c = * curves[ i ];
      Index l = c. getSize();
      tnlStaticVector< 2, Index > m1 = c[ 0 ];
      tnlStaticVector< 2, Index > m2 = c[ l - 1 ];
      Index n1 = abs( ( Index ) m1. x() - ( Index ) m2. x() );
      Index n2 = abs( ( Index ) m1. y() - ( Index ) m2. y() );
      if( ( n1 == 1 && n2 == 0 ) ||
          ( n1 == 0 && n2 == 1 ) )
      {
         dbgCout( "Closing curve no. " << i );
         c. Append( m1 );
      }
   }
 
   // Count precisly the curves points
   const Real a_x = u. getDomainLowerCorner(). x();
   const Real a_y = u. getDomainLowerCorner(). y();
   const Real h_x = u. getSpaceSteps(). x();
   const Real h_y = u. getSpaceSteps(). y();
   tnlStaticVector< 2, Real > null_vector;
   if( ! crv. isEmpty() )
      crv. Append( null_vector, true ); //separator
   for( Index i = 0; i < curves. getSize(); i ++ )
   {
      if( i > 0 ) crv. Append( null_vector, true );  //separator
      tnlList< tnlStaticVector< 2, Index > >& c = * curves[ i ];
      Index l = c. getSize();
      tnlStaticVector< 2, Real > first;
      for( Index j = 0; j < l - 1; j ++ )
      {
         tnlStaticVector< 2, Index > m1 = c[ j ];
         tnlStaticVector< 2, Index > m2 = c[ j + 1 ];
         Index n1 = m2. x() - m1. x();
         Index n2 = m2. y() - m1. y();
         Real p[ 2 ], v[ 2 ];
         if( n1 == 0 && n2 == -1 )
         {
            p[ 0 ] = a_x + ( Real ) m1. x() * h_x;
            p[ 1 ] = a_y + ( Real ) m1. y() * h_y;
            v[ 0 ] = u. getElement( m1. x(), m1. y() ) - level;
            v[ 1 ] = u. getElement( m1. x() + 1, m1. y() ) - level;
         }
         if( n1 == 1 && n2 == 0 )
         {
            p[ 0 ] = a_x + ( Real ) ( m1. x() + 1 ) * h_x;
            p[ 1 ] = a_y + ( Real ) ( m1. y() ) * h_y;
            v[ 0 ] = u. getElement( m1. x() + 1, m1. y() ) - level;
            v[ 1 ] = u. getElement( m1. x() + 1, m1. y() + 1 ) - level;
         }
         if( n1 == 0 && n2 == 1 )
         {
            p[ 0 ] = a_x + ( Real ) ( m1. x() + 1 ) * h_x;
            p[ 1 ] = a_y + ( Real ) ( m1. y() + 1 ) * h_y;
            v[ 0 ] = u. getElement( m1. x() + 1, m1. y() + 1 ) - level;
            v[ 1 ] = u. getElement( m1. x(), m1. y() + 1 ) - level;
         }
         if( n1 == -1 && n2 == 0 )
         {
            p[ 0 ] = a_x + ( Real ) ( m1. x() ) * h_x;
            p[ 1 ] = a_y + ( Real ) ( m1. y() + 1 ) * h_y;
            v[ 0 ] = u. getElement( m1. x(), m1. y() + 1 ) - level;
            v[ 1 ] = u. getElement( m1. x(), m1. y() ) - level;
         }
         if( v[ 0 ] * v[ 1 ] > Real( 0.0 ) )
         {
            // this can happen near the edge or when two curves cross each other
            //curve. Append( CurveElement( Vector2D( 0.0, 0.0 ), true ) );
            //cerr << "Warning: v1 * v2 > 0 in curve identification." << endl;
            continue;
         }
         Real r = v[ 0 ] / ( v[ 1 ] - v[ 0 ] );
         p[ 0 ] += ( ( Real ) n2  ) * r * h_x;
         p[ 1 ] += ( ( Real ) -n1  ) * r * h_y;
         crv. Append( tnlStaticVector< 2, Real >( p ) );
         if( j == 0 ) first = tnlStaticVector< 2, Real >( p );
      }
      tnlStaticVector< 2, Index > m1 = c[ 0 ];
      tnlStaticVector< 2, Index > m2 = c[ l - 1 ];
      if( m1 == m2  )
         crv. Append( first );
   }

   curves. DeepEraseAll();
   return true;
};
 

#endif
