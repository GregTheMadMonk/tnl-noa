/***************************************************************************
                          mString.cpp  -  description
                             -------------------
    begin                : 2004/04/10 16:36
    copyright            : (C) 2004 by Tomas Oberhuber
    email                : tomasoberhuber@seznam.cz
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
#include <config.h>
#include <assert.h>
#include "mString.h"
#include "debug.h"
#include "mfuncs.h"
#ifdef HAVE_MPI
   #include <mpi.h>
#endif

const unsigned int STRING_PAGE = 256;

//---------------------------------------------------------------------------
mString :: mString()
{
   string = new char[ STRING_PAGE ];
   string[ 0 ] = 0;
   length = STRING_PAGE;
}
//---------------------------------------------------------------------------
mString :: mString( const char* c, int prefix_cut_off, int sufix_cut_off )
   : string( 0 ), length( 0 )
{
   SetString( c, prefix_cut_off, sufix_cut_off );
}
//---------------------------------------------------------------------------
mString :: mString( const mString& str ) : length( str. length )
{
   string = new char[ length ];
   unsigned int _length = strlen( str. string );
   memcpy( string, str. string, sizeof( char ) * _length );
   string[ _length ] = 0;
}
//---------------------------------------------------------------------------
mString :: ~mString()
{
   if( string ) delete[] string;
}
//---------------------------------------------------------------------------
void mString :: SetString( const char* c, int prefix_cut_off, int sufix_cut_off )
{
   if( ! c )
   {
      if( ! string )
      {
         string = new char[ STRING_PAGE ];
         length = STRING_PAGE;
      }
      string[ 0 ] = 0;
      return;
   }
   int c_len = ( int ) strlen( c );
   int _length = Max( 0, c_len - prefix_cut_off - sufix_cut_off );
   //assert( _length );
   //dbgExpr( _length );
   //dbgExpr( string );

   if( length < _length || length == 0 )
   {
      if( string ) delete[] string;
      length = STRING_PAGE * ( _length / STRING_PAGE + 1 );
      string = new char[ length ];
   }
   assert( string );
   //dbgExpr( length );
   memcpy( string, c + Min( c_len, prefix_cut_off ), sizeof( char ) * ( _length ) );
   string[ _length ] = 0;
}
//---------------------------------------------------------------------------
mString& mString :: operator = ( const mString& str )
{
   SetString( str. Data() );
   return * this;
}
//---------------------------------------------------------------------------
mString& mString :: operator += ( const char* str )
{
   if( str )
   {
      int len1 = strlen( string );
      int len2 = strlen( str );
      if( len1 + len2 < length )
         memcpy( string + len1, str, sizeof( char ) * ( len2 + 1 ) );
      else
      {
         char* tmp_string = string;
         length = STRING_PAGE * ( ( len1 + len2 ) / STRING_PAGE + 1 );
         string = new char[ length ];
         memcpy( string, tmp_string, sizeof( char ) * len1 );
         memcpy( string + len1, str, sizeof( char ) * ( len2 + 1 ) );
      }
   }
   return * this;
}
//---------------------------------------------------------------------------
mString& mString :: operator += ( const mString& str )
{
   return operator += ( str. Data() );
   /*int len1 = strlen( string );
   int len2 = strlen( str. string );
   if( len1 + len2 < length )
      memcpy( string + len1, str. string, sizeof( char ) * ( len2 + 1 ) );
   else
   {
      char* tmp_string = string;
      length = STRING_PAGE * ( ( len1 + len2 ) / STRING_PAGE + 1 );
      string = new char[ length ];
      memcpy( string, tmp_string, sizeof( char ) * len1 );
      memcpy( string + len1, str. string, sizeof( char ) * ( len2 + 1 ) );
   }
   return * this;*/
}
//---------------------------------------------------------------------------
mString mString :: operator + ( const mString& str )
{
   return mString( *this ) += str;
}
//---------------------------------------------------------------------------
bool mString :: operator == ( const mString& str ) const
{
   assert( string && str. string );
   if( strcmp( string, str. string ) == 0 )
      return true;
   return false;
}
//---------------------------------------------------------------------------
bool mString :: operator != ( const mString& str ) const
{
   return ! operator == ( str );
}
//---------------------------------------------------------------------------
bool mString :: operator == ( const char* str ) const
{
   //cout << ( void* ) string << " " << ( void* ) str << endl;
   assert( string && str );
   if( strcmp( string, str ) == 0 ) return true;
   return false;
}
//---------------------------------------------------------------------------
mString :: operator bool () const
{
   if( string[ 0 ] ) return true;
   return false;
}
//---------------------------------------------------------------------------
bool mString :: operator != ( const char* str ) const
{
   return ! operator == ( str );
}
//---------------------------------------------------------------------------
int mString :: Length() const
{
   return strlen( string );
}
//---------------------------------------------------------------------------
const char* mString :: Data() const
{
   return string;
}
//---------------------------------------------------------------------------
bool mString :: Save( ostream& file ) const
{
   dbgFunctionName( "mString", "Write" );
   assert( string );
   dbgExpr( string );

   int len = strlen( string );
   file. write( ( char* ) &len, sizeof( int ) );
   file. write( string, len );
   if( file. bad() ) return false;
   return true;
}
//---------------------------------------------------------------------------
bool mString :: Load( istream& file )
{
   int _length;
   file. read( ( char* ) &_length, sizeof( int ) );
   if( file. bad() ) return false;
   if( ! _length )
   {
      string[ 0 ] = 0;
      length = 0;
      return true;
   }
   if( string && length < _length )
   {
      delete[] string;
      string = NULL;
   }
   if( ! string ) 
   {
      //dbgCout( "Reallocating string..." );
      length = STRING_PAGE * ( _length / STRING_PAGE + 1 );
      string = new char[ length ];
   }

   file. read( string, _length );
   if( file. bad() ) return false;
   string[ _length ] = 0;
   return true;
}
//---------------------------------------------------------------------------
void mString :: MPIBcast( int root, MPI_Comm comm )
{
#ifdef HAVE_MPI
   dbgFunctionName( "mString", "MPIBcast" );
   int iproc;
   MPI_Comm_rank( MPI_COMM_WORLD, &iproc );
   assert( string );
   int len = strlen( string );
   MPI_Bcast( &len, 1, MPI_INT, root, comm );
   dbgExpr( iproc );
   dbgExpr( len );
   if( iproc != root )
   {
      if( length < len )
      {
         delete[] string;
         length = STRING_PAGE * ( len / STRING_PAGE + 1 );
         string = new char[ length ];
      }
   }
   
   MPI_Bcast( string, len + 1, MPI_CHAR, root, comm );  
   dbgExpr( iproc );
   dbgExpr( string );
#endif
}
//---------------------------------------------------------------------------
ostream& operator << ( ostream& stream, const mString& str )
{
   stream << str. Data();
   return stream;
}
