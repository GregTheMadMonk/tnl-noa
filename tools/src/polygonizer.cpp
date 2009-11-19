/***************************************************************************
                          polygonizer.cpp  -  description
                             -------------------
    begin                : Mon Feb 11 2002
    copyright            : (C) 2002 by Tomas Oberhuber
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

#include <string.h>
#include "polygonizer.h"
#include "../solids/reciever.h"
#include "../solids/solid.h"
#include "hash.h"

POLYGONIZER polygonizer;

static int corner1[ 12 ] =
   { LBN, LTN, LBN, LBF, RBN, RTN, RBN, RBF, LBN, LBF, LTN, LTF };

static int corner2[ 12 ] =
   { LBF, LTF, LTN, LTF, RBF, RTF, RTN, RTF, RBN, RBF, RTN, RTF };

// these are fields of corners for edges in order
//   LB,  LT,  LN,  LF,  RB,  RT,  RN,  RF,  BN,  BF,  TN,  TF

static int left_face[ 12 ] =
   { B, L, L, F, R, T, N, R, N, B, T, F };
// face on left when going corner1 to corner2

static int right_face[ 12 ] =
   { L, T, N, L, B, R, R, F, B, F, N, T };
// face on right when going coner1 to corner2

//---------------------------------------------------------------------------
CUBE :: CUBE( int _i, int _j, int _k )
      : i( _i ), j( _j ), k( _k )
{
   bzero( corners, 8 * sizeof( CORNER* ) );
}
//---------------------------------------------------------------------------
POLYGONIZER :: POLYGONIZER()
{
   edge_hash = new ( list< EDGE* >* )[ 2 * hash_size ];
   bzero( edge_hash, 2 * hash_size * sizeof( list< EDGE* >* ) );
   corner_hash = new ( list< CORNER* >* )[ 2 * hash_size ];
   bzero( corner_hash, 2 * hash_size * sizeof( list< CORNER* >* ) );
   center_hash = new ( list< CENTER* >* )[ 2 * hash_size ];
   bzero( center_hash, 2 * hash_size * sizeof( list< CENTER* >* ) );
   Make_Cube_Table();
}
//---------------------------------------------------------------------------
POLYGONIZER :: ~POLYGONIZER()
{
   assert( edge_hash );
   delete[] edge_hash;
   assert( corner_hash );
   delete[] corner_hash;
   assert( center_hash );
   delete[] center_hash;
}
//---------------------------------------------------------------------------
int POLYGONIZER :: Parametric( const SOLID* sld, RECIEVER* reciever,
                               const unsigned int slices, const unsigned int stacks)
{
   assert( slices && stacks );
   const double x_step = 1.0 / slices;
   const double y_step = 1.0 / stacks;
   for( unsigned int i = 0; i < stacks; i ++ )
      for( unsigned int j = 0; j < slices; j ++ )
      {
         double x = j * x_step;
         double y = i * y_step;
         VECTOR n1, n2, n3, n4;
         VECTOR u1 = sld -> Surface_Point( x, y, &n1 );
         VECTOR u2 = sld -> Surface_Point( x + x_step, y, &n2 );
         VECTOR u3 = sld -> Surface_Point( x + x_step, y + y_step, &n3 );
         VECTOR u4 = sld -> Surface_Point( x, y + y_step, &n4 );

         VERTEX v1( u1, n1 );
         VERTEX v2( u2, n2 );
         VERTEX v3( u3, n3 );
         VERTEX v4( u4, n4 );

         /*TRIANGLE* t1 = new TRIANGLE( new VERTEX( v1 ),
                                      new VERTEX( v4 ),
                                      new VERTEX( v2 ) ) ;
         TRIANGLE* t2 = new TRIANGLE( new VERTEX( v3 ),
                                      new VERTEX( v2 ),
                                      new VERTEX( v4 ) ) ;*/

          if( ! reciever -> Insert_Triangle( v1, v4, v2 ) ) return 0;
          if( ! reciever -> Insert_Triangle( v3, v2, v4 ) ) return 0;
      }
   return 1;
}
//---------------------------------------------------------------------------
int POLYGONIZER :: Init( const VECTOR& pos, const SOLID* sld, const double& cb_sz )
{
   cube_size = cb_sz;
   solid = sld;
   VECTOR in( pos ), out( pos );
   if( ! Find_Point( 1, in, 1.0, solid ) ||
       ! Find_Point( 0, out, 1.0, solid ) ) return 0;
   CUBE *c = NULL;
   if( cube_stack. empty() ) // this is the initial cube
   {
      position = Compute_Surface_Point( in, out, solid -> Continuous_Function( in ) );
      c = new CUBE( 0, 0, 0 );
   }
   else
   {
      VECTOR _pos = Compute_Surface_Point( in, out, solid -> Continuous_Function( in ) ) - position;
      _pos *= 1.0 / cube_size;
      int i( ( int ) _pos. x ),
          j( ( int ) _pos. y ),
          k( ( int ) _pos. z );
      if( Set_Center( i, j, k ) ) return 1; //we have already found this cube
      c = new CUBE( i, j, k );
   }
   for( int i = 0; i < 8; i ++ ) c -> corners[ i ] =
      Set_Corner( ( i >> 2 ) & 1, ( i >> 1 ) & 1, i & 1 );

   cube_stack. push( c );
   return 1;
}
//---------------------------------------------------------------------------
int POLYGONIZER :: Implicit( const SOLID* sld, RECIEVER* _reciever,
                             const VECTOR& ps, const VECTOR& cr,
                             const double& cb_size, int mode )
{
   solid = sld;
   reciever = _reciever;
   cube_size = cb_size;
   bound_ps = ps;
   bound_cr = cr;
   if( cube_stack. empty() && ! Init( solid -> Position(), solid, cb_size ) )
   {
      cerr << "Can not find initial points for polygonization" << endl;
      return 0;
   }
   cout << "Starting polygonizer ... " << endl;
   /*VECTOR in( sld -> Position() ), out( sld -> Position() );
   if( ! Find_Point( 1, in, 1.0 ) ||
       ! Find_Point( 0, out, 1.0 ) )
      {
         cerr << "Can not find initial points for polygonization" << endl;
         return 0;
      }
   cout << "Starting polygonizer ... " << endl;
   position = Compute_Surface_Point( in, out, solid -> Continuous_Function( in ) );

   CUBE *c = new CUBE( 0, 0, 0 );
   for( int i = 0; i < 8; i ++ ) c -> corners[ i ] =
      Set_Corner( ( i >> 2 ) & 1, ( i >> 1 ) & 1, i & 1 );

   cube_stack. push( c );*/
   int jkl;
   CUBE* c;
   while( ! cube_stack. empty() )
   {
      c = cube_stack. top();
      cout << "Current stack size is " << cube_stack. size() << " cubes. " << '\r' << flush;
      cube_stack. pop();
      jkl = cube_stack. size();
      if( ! Triangulate_Cube( c ) )
      {
         Free_Memory();
         return 0;
      }
      Test_Face( c -> i - 1, c -> j, c -> k, c, L, LBN, LBF, LTN, LTF );
      cout << "Current stack size is " << cube_stack. size() << " cubes. " << '\r' << flush;
      Test_Face( c -> i + 1, c -> j, c -> k, c, R, RBN, RBF, RTN, RTF );
      cout << "Current stack size is " << cube_stack. size() << " cubes. " << '\r' << flush;
      Test_Face( c -> i, c -> j - 1, c -> k, c, B, LBN, LBF, RBN, RBF );
      cout << "Current stack size is " << cube_stack. size() << " cubes. " << '\r' << flush;
      Test_Face( c -> i, c -> j + 1, c -> k, c, T, LTN, LTF, RTN, RTF );
      cout << "Current stack size is " << cube_stack. size() << " cubes. " << '\r' << flush;
      Test_Face( c -> i, c -> j, c -> k - 1, c, N, LBN, LTN, RBN, RTN );
      cout << "Current stack size is " << cube_stack. size() << " cubes. " << '\r' << flush;
      Test_Face( c -> i, c -> j, c -> k + 1, c, F, LBF, LTF, RBF, RTF );
      cout << "Current stack size is " << cube_stack. size() << " cubes. " << '\r' << flush;
   }
   cout << endl;
   Free_Memory();
   return 1;
}
//---------------------------------------------------------------------------
int POLYGONIZER :: Triangulate_Cube( CUBE* cube )
{
   #ifdef DBG_POLYGONIZER
   cout << "Cube polygonize: " << cube -> i << " " << cube -> j << " " << cube -> k << endl;;
   #endif
   int index = 0;
   // this is a index of cube in cube_table, it will be count by
   // the signs of function in corners
   for( int i = 0; i < 8; i ++ )
      if( cube -> corners[ i ] -> value > 0.0 ) index += ( 1 << i );
   for( size_t i = 0; i < cube_table[ index ]. size(); i ++ )
   {
      vector< int >& edges_vector = ( cube_table[ index ] )[ i ];
      VERTEX *v1( 0 ), *v2( 0 ), *v3( 0 );
      for( size_t j = 0; j < edges_vector. size(); j ++ )
      {
         CORNER* c1 = cube -> corners[ corner1[ edges_vector[ j ] ] ];
         CORNER* c2 = cube -> corners[ corner2[ edges_vector[ j ] ] ];
         VERTEX* c = Get_Vertex( c1, c2 );
         if( j == 0 ) v1 = c;
         if( j == 1 ) v2 = c;
         if( j > 1 )
         {
            v3 = c;
            if( ! reciever -> Insert_Triangle( *v1, *v3, *v2 ) ) return 0;
            v2 = v3;
         }
      }
   }
   return 1;
}
//---------------------------------------------------------------------------
VERTEX* POLYGONIZER :: Get_Vertex( CORNER* c1, CORNER* c2 )
{
   VERTEX* vertex;
   if( Check_Edge( c1, c2, vertex ) ) return vertex;
   // the vertex has been already computed
   VECTOR p = Compute_Surface_Point( c1 -> position, c2 -> position, c1 -> value );
   vertex = new VERTEX( p, solid -> Normal( p ) );
   Set_Edge( c1, c2, vertex );
   return vertex;
}
//---------------------------------------------------------------------------
VECTOR POLYGONIZER :: Compute_Surface_Point( const VECTOR& v1, const VECTOR& v2, const double& val )
{
   VECTOR pos, neg, p;
   if( val < 0.0 )
   {
      pos = v2; neg = v1;
   }
   else
   {
      pos = v1; neg = v2;
   }
   int i = 0;
   while( i ++ < max_iter )
   {
      p = 0.5 * ( pos + neg );
      if( solid -> Continuous_Function( p ) > 0 ) pos = p;
      else neg = p;
   }
   return p;
}
//---------------------------------------------------------------------------
void POLYGONIZER :: Set_Edge( CORNER* c1, CORNER* c2, VERTEX* v )
{
   int& i1 = c1 -> i;
   int& j1 = c1 -> j;
   int& k1 = c1 -> k;
   int& i2 = c2 -> i;
   int& j2 = c2 -> j;
   int& k2 = c2 -> k;
   if( i1 > i2 || ( i1 == i2 && ( j1 > j2 || ( j1 == j2  && k1 > k2 ) ) ) )
   {
      CORNER* c = c1; c1 = c2; c2 = c;
   }
   unsigned int index = hash_index( i1, j1, k1 ) + hash_index( i2, j2, k2 );
   if( ! edge_hash[ index ] ) edge_hash[ index ] = new list< EDGE* >;
   edge_hash[ index ] -> push_back( new EDGE( c1, c2, v ) );
}
//---------------------------------------------------------------------------
int POLYGONIZER :: Check_Edge( CORNER* c1, CORNER* c2, VERTEX*& vertex )
{
   int& i1 = c1 -> i;
   int& j1 = c1 -> j;
   int& k1 = c1 -> k;
   int& i2 = c2 -> i;
   int& j2 = c2 -> j;
   int& k2 = c2 -> k;
   if( i1 > i2 || ( i1 == i2 && ( j1 > j2 || ( j1 == j2  && k1 > k2 ) ) ) )
   {
      CORNER* c = c1; c1 = c2; c2 = c;
   }
   list< EDGE* >* el = edge_hash[ hash_index( i1, j1, k1 ) + hash_index( i2, j2, k2 ) ];
   if( ! el ) return 0;
   list< EDGE* > :: iterator it = el -> begin();
   while( it != el -> end() )
   {
      EDGE* e = * it ++;
      if( e -> corners[ 0 ] == c1 && e -> corners[ 1 ] == c2 )
      {
         vertex = e -> vertex;
         return 1;
      }
   }
   return 0;
}
//---------------------------------------------------------------------------
CORNER* POLYGONIZER :: Set_Corner( int i, int j, int k )
{
   #ifdef DBG_POLYGONIZER
      cout << "Set Corner index: " << i << " " << j << " " << k << endl;
   #endif
   int index = hash_index( i, j, k );
   list< CORNER* >*& cl = corner_hash[ index ];
   if( ! cl ) cl = new list< CORNER* >;
   list< CORNER* > :: iterator it = cl -> begin();
   CORNER* cr;
   while( it != cl -> end() )
   {
      cr = * it ++;
      if( ( cr -> i == i ) && ( cr -> j == j ) && ( cr -> k == k ) ) return cr;
   }
   VECTOR tmp = ( VECTOR( i, j, k ) - VECTOR( 0.5 ) );
   VECTOR p = position + cube_size * tmp;
   double val = solid -> Continuous_Function( p );
   #ifdef DBG_POLYGONIZER
   cout << "Set Corer: value " << val << endl;
   #endif
   cr = new CORNER( i, j, k, p, val );
   cl -> push_back( cr );
   return cr;
}
//---------------------------------------------------------------------------
int POLYGONIZER :: Set_Center( int i, int j, int k )
{
   #ifdef DBG_POLYGONIZER
   cout << "Set Center: " << i << " " << j << " " << k << endl;
   #endif
   list< CENTER* >*& cl = center_hash[ hash_index( i, j, k ) ];
   if( ! cl ) cl = new list< CENTER* >;
   list< CENTER* > :: iterator it = cl -> begin();
   while( it != cl -> end() )
   {
      CENTER* cr = * it ++;
      if( cr -> i == i && cr -> j == j && cr -> k == k ) return 1;
   }
   cl -> push_back( new CENTER( i, j, k ) );
   return 0;
}
//---------------------------------------------------------------------------
void POLYGONIZER :: Make_Cube_Table()
{
   int done_edges[ 12 ], positive[ 8 ];
   // we have 256 different possibilities to sign all 8 corners of a cube
   // they are describe here in cube_table
   for( int i = 0; i < 256; i ++ )
   {
      for( int e = 0; e < 12; e ++ ) done_edges[ e ] = 0;
      for( int c = 0; c < 8; c ++ ) positive[ c ] = ( i >> c ) & 1;
      // each bit in 'i' represent one corner of cube and sign of function value
      // '1' means positive value, '0' means negative value
      for( int e = 0; e < 12; e ++ )
      {
         if( ! done_edges[ e ] &&
             // we didn't process this edge yet ...
             ( positive[ corner1[ e ] ] != positive[ corner2[ e ] ] ) )
             // ... and corners of this edge have different sign
            {
               int start = e;
               int edge = e;
               int face = positive[ corner1[ e ] ] ? right_face[ e ] : left_face[ e ];
               // get face that is to right of edge from positive to negatve corner
               vector< int > temp_vector;
               while( 1 )
               {
                  edge = Next_Clockwise_Edge( edge, face );
                  done_edges[ edge ] = 1;
                  if( positive[ corner1[ edge ] ] != positive[ corner2[ edge ] ] )
                  {
                     temp_vector. push_back( edge );
                     if( edge == start ) break;
                     face = Other_Face( edge, face );
                  }
               }
               cube_table[ i ]. push_back( temp_vector );
            }
      }
   }
}
//---------------------------------------------------------------------------
int POLYGONIZER :: Next_Clockwise_Edge( int edge, int face )
{
   switch( edge )
   {
      case LB: return ( face == L ) ? LF : BN;
      case LT: return ( face == L ) ? LN : TF;
      case LN: return ( face == L ) ? LB : TN;
      case LF: return ( face == L ) ? LT : BF;
      case RB: return ( face == R ) ? RN : BF;
      case RT: return ( face == R ) ? RF : TN;
      case RN: return ( face == R ) ? RT : BN;
      case RF: return ( face == R ) ? RB : TF;
      case BN: return ( face == B ) ? RB : LN;
      case BF: return ( face == B ) ? LB : RF;
      case TN: return ( face == T ) ? LT : RN;
      case TF: return ( face == T ) ? RT : LF;
   }
   return 0; // this is just for avoiding compiler warning
}
//---------------------------------------------------------------------------
int POLYGONIZER :: Other_Face( int edge, int face )
{
   int other_face = left_face[ edge ];
   return face == other_face ? right_face[ edge ] : other_face;
}
//---------------------------------------------------------------------------
void POLYGONIZER :: Test_Face( int i, int j, int k, CUBE* old, int face,
                               int c1, int c2, int c3, int c4 )
{
   static int face_bit[ 6 ] = { 2, 2, 1, 1, 0, 0 };
   int bit = face_bit[ face ];
   int pos = old -> corners[ c1 ] -> value > 0.0 ? 1 : 0;
   // test id no surface crossing, cube out of bounds, or already visited
   if( ( old -> corners[ c2 ] -> value > 0.0 ) == pos &&
       ( old -> corners[ c3 ] -> value > 0.0 ) == pos &&
       ( old -> corners[ c4 ] -> value > 0.0 ) == pos ) return;
   // test bounds
   if( bound_ps < bound_cr )
   {
      VECTOR p( position + cube_size * VECTOR( ( double ) i, ( double ) j, ( double ) k ) );
      if( ! ( p >= bound_ps && p <= bound_cr ) ) return;
   }
   else
      if( abs( i ) > 50 || abs( j ) > 50 || abs( k ) > 50 ) return;
   if( Set_Center( i, j, k ) ) return;
   CUBE* new_cube = new CUBE( i, j, k );
   // given face of old cube is the same as the opposite face of new_cube
   new_cube -> corners[ flip_bit( c1, bit ) ] = old -> corners[ c1 ];
   new_cube -> corners[ flip_bit( c2, bit ) ] = old -> corners[ c2 ];
   new_cube -> corners[ flip_bit( c3, bit ) ] = old -> corners[ c3 ];
   new_cube -> corners[ flip_bit( c4, bit ) ] = old -> corners[ c4 ];
   #ifdef DBG_POLYGONIZER
   cout << "Test Face indexes: " << i << " " << j << " " << k << endl;
   #endif
   for( int n = 0; n < 8; n ++ )
      if( ! new_cube -> corners[ n ] )
         new_cube -> corners[ n ] = Set_Corner( i + ( ( n >> 2 ) & 1 ),
                                                j + ( ( n >> 1 ) & 1 ),
                                                k + ( n & 1 ) );
   cube_stack. push( new_cube );
}
//---------------------------------------------------------------------------
int POLYGONIZER :: Find_Point( int sign, VECTOR& point, double size, const SOLID* sld )
{
   VECTOR init = point;
   VECTOR ps( -0.5 );
   VECTOR cr(  0.5 );
   for( int i = 0; i < 10000; i ++ )
   {
      point = init + size * Random_Vector( ps, cr );
      //cout << point << " - " << sld -> Continuous_Function( point ) << endl;
      if( sign == ( sld -> Sign_Function( point ) == 1 ) )
         return 1;
      size *= 1.005;
   }
   return 0;
}
//---------------------------------------------------------------------------
void POLYGONIZER :: Free_Memory()
{
   // clear hashes
   for( size_t i = 0; i < 2 * hash_size; i ++ )
   {
      if( edge_hash[ i ] )
      {
         list< EDGE* > :: iterator it = edge_hash[ i ] -> begin();
         while( it != edge_hash[ i ] -> end() ) delete * it ++;
         delete edge_hash[ i ];
         edge_hash[ i ] = NULL;

      }

      if( corner_hash[ i ] )
      {
         list< CORNER* > :: iterator it = corner_hash[ i ] -> begin();
         while( it != corner_hash[ i ] -> end() ) delete * it ++;
         delete corner_hash[ i ];
         corner_hash[ i ] = NULL;
      }
      if( center_hash[ i ] )
      {
         list< CENTER* > :: iterator it = center_hash[ i ] -> begin();
         while( it != center_hash[ i ] -> end() ) delete * it ++;
         delete center_hash[ i ];
         center_hash[ i ] = NULL;
      }
   }
}

