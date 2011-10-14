/***************************************************************************
                          tnlString.cpp  -  description
                             -------------------
    begin                : 2004/04/10 16:36
    copyright            : (C) 2004 by Tomas Oberhuber
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

#include <cstring>
#include <string.h>
#include <core/tnlString.h>
#include <debug/tnlDebug.h>
#include <core/tnlAssert.h>
#include <core/tnlList.h>
#include <core/tnlFile.h>
#include "mfuncs.h"
#ifdef HAVE_MPI
   #include <mpi.h>
#endif

const unsigned int STRING_PAGE = 256;

tnlString :: tnlString()
{
   string = new char[ STRING_PAGE ];
   string[ 0 ] = 0;
   length = STRING_PAGE;
}

tnlString :: tnlString( const char* c, int prefix_cut_off, int sufix_cut_off )
   : string( 0 ), length( 0 )
{
   setString( c, prefix_cut_off, sufix_cut_off );
}

tnlString :: tnlString( const tnlString& str )
: string( 0 ), length( 0 )
{
   setString( str. getString() );
}

tnlString :: tnlString( int number )
{
   string = new char[ STRING_PAGE ];
   length = STRING_PAGE;
   sprintf( string, "%d", number );
}

tnlString :: ~tnlString()
{
   if( string ) delete[] string;
}
//---------------------------------------------------------------------------
void tnlString :: setString( const char* c, int prefix_cut_off, int sufix_cut_off )
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
const char& tnlString :: operator[]( int i ) const
{
   tnlAssert( i >= 0 && i < length,
              cerr << "Accessing char outside the string." );
   return string[ i ];
}
//---------------------------------------------------------------------------
char& tnlString :: operator[]( int i )
{
   tnlAssert( i >= 0 && i < length,
              cerr << "Accessing char outside the string." );
   return string[ i ];
}
//---------------------------------------------------------------------------
tnlString& tnlString :: operator = ( const tnlString& str )
{
   setString( str. getString() );
   return * this;
}
//---------------------------------------------------------------------------
tnlString& tnlString :: operator += ( const char* str )
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
tnlString& tnlString :: operator += ( const char str )
{
   int len1 = strlen( string );
   if( len1 + 1 < length )
   {
      string[ len1 ] = str;
      string[ len1 + 1 ] = 0;
   }
   else
   {
      char* tmp_string = string;
      length += STRING_PAGE;
      string = new char[ length ];
      memcpy( string, tmp_string, sizeof( char ) * len1 );
      string[ len1 ] = str;
      string[ len1 + 1 ] = 0;
   }

   return * this;
}
//---------------------------------------------------------------------------
tnlString& tnlString :: operator += ( const tnlString& str )
{
   return operator += ( str. getString() );
}
//---------------------------------------------------------------------------
tnlString tnlString :: operator + ( const tnlString& str )
{
   return tnlString( *this ) += str;
}
//---------------------------------------------------------------------------
tnlString tnlString :: operator + ( const char* str )
{
   return tnlString( *this ) += str;
}
//---------------------------------------------------------------------------
bool tnlString :: operator == ( const tnlString& str ) const
{
   assert( string && str. string );
   if( str. length != length )
      return false;
   if( strcmp( string, str. string ) == 0 )
      return true;
   return false;
}
//---------------------------------------------------------------------------
bool tnlString :: operator != ( const tnlString& str ) const
{
   return ! operator == ( str );
}
//---------------------------------------------------------------------------
bool tnlString :: operator == ( const char* str ) const
{
   //cout << ( void* ) string << " " << ( void* ) str << endl;
   assert( string && str );
   if( strcmp( string, str ) == 0 ) return true;
   return false;
}
//---------------------------------------------------------------------------
tnlString :: operator bool () const
{
   if( string[ 0 ] ) return true;
   return false;
}
//---------------------------------------------------------------------------
bool tnlString :: operator != ( const char* str ) const
{
   return ! operator == ( str );
}
//---------------------------------------------------------------------------
int tnlString :: getLength() const
{
   return strlen( string );
}
//---------------------------------------------------------------------------
const char* tnlString :: getString() const
{
   return string;
}
//---------------------------------------------------------------------------
char* tnlString :: getString()
{
   return string;
}

//---------------------------------------------------------------------------
bool tnlString :: save( ostream& file ) const
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
bool tnlString :: load( istream& file )
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
bool tnlString :: save( tnlFile& file ) const
{
   dbgFunctionName( "tnlString", "Write" );
   tnlAssert( string,
              cerr << "string = " << string );
   dbgExpr( string );

   int len = strlen( string );
   if( ! file. write( &len, 1 ) )
      return false;
   if( ! file. write( string, len ) )
      return false;
   return true;
}
//---------------------------------------------------------------------------
bool tnlString :: load( tnlFile& file )
{
   int _length;
   if( ! file. read( &_length, 1 ) )
   {
      cerr << "I was not able to read tnlString length." << endl;
      return false;
   }
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

   if( ! file. read( string, _length ) )
   {
      cerr << "I was not able to read a tnlString with a length " << length << "." << endl;
      return false;
   }
   string[ _length ] = 0;
   return true;
}
//---------------------------------------------------------------------------
void tnlString :: MPIBcast( int root, MPI_Comm comm )
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
bool tnlString :: getLine( istream& stream )
{
   std :: string str;
   getline( stream, str );
   this -> setString( str. data() );
   if( ! ( *this ) ) return false;
   return true;
}
//---------------------------------------------------------------------------
int tnlString :: parse( tnlList< tnlString >& list, const char separator ) const
{
   dbgFunctionName( "tnlString", "parse" );
   list. EraseAll();
   tnlString copy( *this );
   int len = copy. getLength();
   for( int i = 0; i < len; i ++ )
      if( copy[ i ] == separator )
         copy[ i ] = 0;
   for( int i = 0; i < len; i ++ )
   {
      if( copy[ i ] == 0 ) continue;
      tnlString new_string;
      new_string. setString( &copy. getString()[ i ] );
      dbgExpr( new_string );
      i += new_string. getLength();
      list. Append( new_string );
   }
   return list. getSize();
}
//---------------------------------------------------------------------------
ostream& operator << ( ostream& stream, const tnlString& str )
{
   stream << str. getString();
   return stream;
}
