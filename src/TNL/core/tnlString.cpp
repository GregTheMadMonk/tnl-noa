/***************************************************************************
                          tnlString.cpp  -  description
                             -------------------
    begin                : 2004/04/10 16:36
    copyright            : (C) 2004 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <cstring>
#include <string.h>
#include <assert.h>
#include <TNL/core/tnlString.h>
#include <TNL/debug/tnlDebug.h>
#include <TNL/core/tnlAssert.h>
#include <TNL/core/tnlList.h>
#include <TNL/core/tnlFile.h>
#include <TNL/core/mfuncs.h>
#ifdef HAVE_MPI
   #include <mpi.h>
#endif

namespace TNL {

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

tnlString :: tnlString( unsigned number )
: string( 0 ), length( 0 )
{
   this->setString( convertToString( number ).getString() );
}

tnlString :: tnlString( int number )
: string( 0 ), length( 0 )
{
   this->setString( convertToString( number ).getString() );
}

tnlString :: tnlString( long int number )
: string( 0 ), length( 0 )
{
   this->setString( convertToString( number ).getString() );
}

tnlString :: tnlString( float number )
: string( 0 ), length( 0 )
{
   this->setString( convertToString( number ).getString() );
}

tnlString :: tnlString( double number )
: string( 0 ), length( 0 )
{
   this->setString( convertToString( number ).getString() );
}

tnlString tnlString :: getType()
{
   return tnlString( "tnlString" );
}

tnlString :: ~tnlString()
{
   if( string ) delete[] string;
}

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
   int _length = max( 0, c_len - prefix_cut_off - sufix_cut_off );

   if( length < _length || length == 0 )
   {
      if( string ) delete[] string;
      length = STRING_PAGE * ( _length / STRING_PAGE + 1 );
      string = new char[ length ];
   }
   tnlAssert( string, );
   memcpy( string, c + min( c_len, prefix_cut_off ), sizeof( char ) * ( _length ) );
   string[ _length ] = 0;
}

const char& tnlString :: operator[]( int i ) const
{
   tnlAssert( i >= 0 && i < length,
              std::cerr << "Accessing char outside the string." );
   return string[ i ];
}

char& tnlString :: operator[]( int i )
{
   tnlAssert( i >= 0 && i < length,
              std::cerr << "Accessing char outside the string." );
   return string[ i ];
}

tnlString& tnlString :: operator = ( const tnlString& str )
{
   setString( str. getString() );
   return * this;
}

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

tnlString& tnlString :: operator += ( const tnlString& str )
{
   return operator += ( str. getString() );
}

tnlString tnlString :: operator + ( const tnlString& str ) const
{
   return tnlString( *this ) += str;
}

tnlString tnlString :: operator + ( const char* str ) const
{
   return tnlString( *this ) += str;
}

bool tnlString :: operator == ( const tnlString& str ) const
{
   assert( string && str. string );
   if( str. length != length )
      return false;
   if( strcmp( string, str. string ) == 0 )
      return true;
   return false;
}

bool tnlString :: operator != ( const tnlString& str ) const
{
   return ! operator == ( str );
}

bool tnlString :: operator == ( const char* str ) const
{
   //cout << ( void* ) string << " " << ( void* ) str << std::endl;
   assert( string && str );
   if( strcmp( string, str ) == 0 ) return true;
   return false;
}

tnlString :: operator bool () const
{
   if( string[ 0 ] ) return true;
   return false;
}

bool tnlString :: operator != ( const char* str ) const
{
   return ! operator == ( str );
}

int tnlString :: getLength() const
{
   return strlen( string );
}

void
tnlString::
replace( const tnlString& pattern,
         const tnlString& replaceWith )
{
   int occurences( 0 );
   int patternLength = pattern.getLength();
   const int length = this->getLength();
   int patternPointer( 0 );
   for( int i = 0; i < length; i++ )
   {
      if( this->string[ i ] == pattern[ patternPointer ] )
         patternPointer++;
      if( patternPointer == patternLength )
      {
         occurences++;
         patternPointer = 0;
      }
   }
   const int replaceWithLength = replaceWith.getLength();
   int newStringLength = length + occurences * ( replaceWithLength - patternLength );
   char* newString = new char[ newStringLength ];
   int newStringPointer( 0 );
   int lastPatternStart( 0 );
   for( int i = 0; i < length; i++ )
   {
      if( this->string[ i ] == pattern[ patternPointer ] )
      {
         if( patternPointer == 0 )
            lastPatternStart = newStringPointer;
         patternPointer++;
      }
      newString[ newStringPointer++ ] = this->string[ i ];
      if( patternPointer == patternLength )
      {
         newStringPointer = lastPatternStart;
         for( int j = 0; j < replaceWithLength; j++ )
            newString[ newStringPointer++ ] = replaceWith[ j ];
         patternPointer = 0;
      }
   }
   delete[] this->string;
   this->string = newString;
}


const char* tnlString :: getString() const
{
   return string;
}

char* tnlString :: getString()
{
   return string;
}


bool tnlString :: save( std::ostream& file ) const
{
   dbgFunctionName( "tnlString", "save" );
   assert( string );
   dbgExpr( string );

   int len = strlen( string );
   file. write( ( char* ) &len, sizeof( int ) );
   file. write( string, len );
   if( file. bad() ) return false;
   return true;
}

bool tnlString :: load( std::istream& file )
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

bool tnlString :: save( tnlFile& file ) const
{
   dbgFunctionName( "tnlString", "Write" );
   tnlAssert( string,
              std::cerr << "string = " << string );
   dbgExpr( string );

   int len = strlen( string );
#ifdef HAVE_NOT_CXX11
   if( ! file. write< int, tnlHost >( &len ) )
#else
   if( ! file. write( &len ) )
#endif
      return false;
#ifdef HAVE_NOT_CXX11
   if( ! file. write< char, tnlHost, int >( string, len ) )
#else
   if( ! file. write( string, len ) )
#endif
      return false;
   return true;
}

bool tnlString :: load( tnlFile& file )
{
   int _length;
#ifdef HAVE_NOT_CXX11
   if( ! file. read< int, tnlHost >( &_length ) )
#else
   if( ! file. read( &_length ) )
#endif
   {
      std::cerr << "I was not able to read tnlString length." << std::endl;
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

#ifdef HAVE_NOT_CXX11
   if( ! file. read< char, tnlHost, int >( string, _length ) )
#else
   if( ! file. read( string, _length ) )
#endif
   {
      std::cerr << "I was not able to read a tnlString with a length " << length << "." << std::endl;
      return false;
   }
   string[ _length ] = 0;
   return true;
}

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

bool tnlString :: getLine( std::istream& stream )
{
   std :: string str;
   getline( stream, str );
   this->setString( str. data() );
   if( ! ( *this ) ) return false;
   return true;
}

int tnlString :: parse( tnlList< tnlString >& list, const char separator ) const
{
   dbgFunctionName( "tnlString", "parse" );
   list.reset();
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

tnlString operator + ( const char* string1, const tnlString& string2 )
{
   return tnlString( string1 ) + string2;
}

std::ostream& operator << ( std::ostream& stream, const tnlString& str )
{
   stream << str. getString();
   return stream;
}

} // namespace TNL
