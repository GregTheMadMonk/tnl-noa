/***************************************************************************
                          String.cpp  -  description
                             -------------------
    begin                : 2004/04/10 16:36
    copyright            : (C) 2004 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <cstring>
#include <string.h>
#include <assert.h>
#include <TNL/String.h>
#include <TNL/Assert.h>
#include <TNL/Containers/List.h>
#include <TNL/File.h>
#include <TNL/Math.h>
#ifdef HAVE_MPI
   #include <mpi.h>
#endif

namespace TNL {

const unsigned int STRING_PAGE = 256;

String :: String()
{
   string = new char[ STRING_PAGE ];
   string[ 0 ] = 0;
   length = STRING_PAGE;
}

String :: String( const char* c, int prefix_cut_off, int sufix_cut_off )
   : string( 0 ), length( 0 )
{
   setString( c, prefix_cut_off, sufix_cut_off );
}

String :: String( const String& str )
: string( 0 ), length( 0 )
{
   setString( str. getString() );
}

String :: String( unsigned number )
: string( 0 ), length( 0 )
{
   this->setString( convertToString( number ).getString() );
}

String :: String( int number )
: string( 0 ), length( 0 )
{
   this->setString( convertToString( number ).getString() );
}

String :: String( unsigned long int number )
: string( 0 ), length( 0 )
{
   this->setString( convertToString( number ).getString() );
}

String :: String( long int number )
: string( 0 ), length( 0 )
{
   this->setString( convertToString( number ).getString() );
}

String :: String( float number )
: string( 0 ), length( 0 )
{
   this->setString( convertToString( number ).getString() );
}

String :: String( double number )
: string( 0 ), length( 0 )
{
   this->setString( convertToString( number ).getString() );
}

String String :: getType()
{
   return String( "String" );
}

String :: ~String()
{
   if( string ) delete[] string;
}

void String :: setString( const char* c, int prefix_cut_off, int sufix_cut_off )
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
   TNL_ASSERT( string, );
   memcpy( string, c + min( c_len, prefix_cut_off ), sizeof( char ) * ( _length ) );
   string[ _length ] = 0;
}

const char& String :: operator[]( int i ) const
{
   TNL_ASSERT( i >= 0 && i < length,
              std::cerr << "Accessing char outside the string." );
   return string[ i ];
}

char& String :: operator[]( int i )
{
   TNL_ASSERT( i >= 0 && i < length,
              std::cerr << "Accessing char outside the string." );
   return string[ i ];
}

String& String :: operator = ( const String& str )
{
   setString( str. getString() );
   return * this;
}

String& String :: operator += ( const char* str )
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

String& String :: operator += ( const char str )
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

String& String :: operator += ( const String& str )
{
   return operator += ( str. getString() );
}

String String :: operator + ( const String& str ) const
{
   return String( *this ) += str;
}

String String :: operator + ( const char* str ) const
{
   return String( *this ) += str;
}

bool String :: operator == ( const String& str ) const
{
   assert( string && str. string );
   if( str. length != length )
      return false;
   if( strcmp( string, str. string ) == 0 )
      return true;
   return false;
}

bool String :: operator != ( const String& str ) const
{
   return ! operator == ( str );
}

bool String :: operator == ( const char* str ) const
{
   //cout << ( void* ) string << " " << ( void* ) str << std::endl;
   assert( string && str );
   if( strcmp( string, str ) == 0 ) return true;
   return false;
}

String :: operator bool () const
{
   if( string[ 0 ] ) return true;
   return false;
}

bool String :: operator != ( const char* str ) const
{
   return ! operator == ( str );
}

int String :: getLength() const
{
   return strlen( string );
}

void
String::
replace( const String& pattern,
         const String& replaceWith )
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

String
String::strip( char strip ) const
{
   int prefix_cut_off = 0;
   int sufix_cut_off = 0;

   while( prefix_cut_off < getLength() && (*this)[ prefix_cut_off ] == strip )
      prefix_cut_off++;

   while( sufix_cut_off < getLength() && (*this)[ getLength() - 1 - sufix_cut_off ] == strip )
      sufix_cut_off++;

   if( prefix_cut_off + sufix_cut_off < getLength() )
      return String( getString(), prefix_cut_off, sufix_cut_off );
   return "";
}


const char* String :: getString() const
{
   return string;
}

char* String :: getString()
{
   return string;
}


bool String :: save( std::ostream& file ) const
{
   assert( string );

   int len = strlen( string );
   file. write( ( char* ) &len, sizeof( int ) );
   file. write( string, len );
   if( file. bad() ) return false;
   return true;
}

bool String :: load( std::istream& file )
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

bool String :: save( File& file ) const
{
   TNL_ASSERT( string,
              std::cerr << "string = " << string );

   int len = strlen( string );
#ifdef HAVE_NOT_CXX11
   if( ! file. write< int, Devices::Host >( &len ) )
#else
   if( ! file. write( &len ) )
#endif
      return false;
#ifdef HAVE_NOT_CXX11
   if( ! file. write< char, Devices::Host, int >( string, len ) )
#else
   if( ! file. write( string, len ) )
#endif
      return false;
   return true;
}

bool String :: load( File& file )
{
   int _length;
#ifdef HAVE_NOT_CXX11
   if( ! file. read< int, Devices::Host >( &_length ) )
#else
   if( ! file. read( &_length ) )
#endif
   {
      std::cerr << "I was not able to read String length." << std::endl;
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
   if( ! file. read< char, Devices::Host, int >( string, _length ) )
#else
   if( ! file. read( string, _length ) )
#endif
   {
      std::cerr << "I was not able to read a String with a length " << length << "." << std::endl;
      return false;
   }
   string[ _length ] = 0;
   return true;
}

/*void String :: MPIBcast( int root, MPI_Comm comm )
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
*/
bool String :: getLine( std::istream& stream )
{
   std :: string str;
   getline( stream, str );
   this->setString( str. data() );
   if( ! ( *this ) ) return false;
   return true;
}

int String :: parse( Containers::List< String >& list, const char separator ) const
{
   list.reset();
   String copy( *this );
   int len = copy. getLength();
   for( int i = 0; i < len; i ++ )
      if( copy[ i ] == separator )
         copy[ i ] = 0;
   for( int i = 0; i < len; i ++ )
   {
      if( copy[ i ] == 0 ) continue;
      String new_string;
      new_string. setString( &copy. getString()[ i ] );
      i += new_string. getLength();
      list. Append( new_string );
   }
   return list. getSize();
}

String operator + ( const char* string1, const String& string2 )
{
   return String( string1 ) + string2;
}

std::ostream& operator << ( std::ostream& stream, const String& str )
{
   stream << str. getString();
   return stream;
}

} // namespace TNL
