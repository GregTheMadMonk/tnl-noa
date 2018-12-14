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
#include <TNL/String.h>
#include <TNL/Assert.h>
#include <TNL/Containers/List.h>
#include <TNL/File.h>
#include <TNL/Math.h>
#ifdef USE_MPI
   #include <mpi.h>
#endif

namespace TNL {

const unsigned int STRING_PAGE = 256;

String::String()
   : string( nullptr ), length( 0 )
{
   setString( nullptr );
}

String::String( const char* c, int prefix_cut_off, int sufix_cut_off )
   : string( nullptr ), length( 0 )
{
   setString( c, prefix_cut_off, sufix_cut_off );
}

String::String( const String& str )
   : string( nullptr ), length( 0 )
{
   setString( str.getString() );
}

String String::getType()
{
   return String( "String" );
}

String::~String()
{
   if( string ) delete[] string;
}

int String::getLength() const
{
   return getSize();
}

int String::getSize() const
{
   return strlen( string );
}

int String::getAllocatedSize() const
{
   return length;
}

void String::setSize( int size )
{
   TNL_ASSERT_GE( size, 0, "string size must be non-negative" );
   const int _length = STRING_PAGE * ( size / STRING_PAGE + 1 );
   TNL_ASSERT_GE( _length, 0, "_length size must be non-negative" );
   if( length != _length ) {
      if( string ) {
         delete[] string;
         string = nullptr;
      }
      string = new char[ _length ];
      length = _length;
   }
}

void String::setString( const char* c, int prefix_cut_off, int sufix_cut_off )
{
   if( ! c ) {
      if( ! string )
         setSize( 1 );
      string[ 0 ] = 0;
      return;
   }
   const int c_len = ( int ) strlen( c );
   const int _length = max( 0, c_len - prefix_cut_off - sufix_cut_off );

   if( length < _length || length == 0 )
      setSize( _length );
   TNL_ASSERT( string, );
   memcpy( string, c + min( c_len, prefix_cut_off ), _length * sizeof( char ) );
   string[ _length ] = 0;
}

const char* String::getString() const
{
   return string;
}

char* String::getString()
{
   return string;
}


const char& String::operator[]( int i ) const
{
   TNL_ASSERT( i >= 0 && i < length,
               std::cerr << "Accessing char outside the string." );
   return string[ i ];
}

char& String::operator[]( int i )
{
   TNL_ASSERT( i >= 0 && i < length,
               std::cerr << "Accessing char outside the string." );
   return string[ i ];
}


/****
 * Operators for C strings
 */
String& String::operator=( const char* str )
{
   setString( str );
   return *this;
}

String& String::operator+=( const char* str )
{
   if( str )
   {
      const int len1 = strlen( string );
      const int len2 = strlen( str );
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
   return *this;
}

String String::operator+( const char* str ) const
{
   return String( *this ) += str;
}

bool String::operator==( const char* str ) const
{
   TNL_ASSERT( string && str, );
   return strcmp( string, str ) == 0;
}

bool String::operator!=( const char* str ) const
{
   return ! operator==( str );
}


/****
 * Operators for Strings
 */
String& String::operator=( const String& str )
{
   setString( str.getString() );
   return *this;
}

String& String::operator+=( const String& str )
{
   return operator+=( str.getString() );
}

String String::operator+( const String& str ) const
{
   return String( *this ) += str;
}

bool String::operator==( const String& str ) const
{
   TNL_ASSERT( string && str.string, );
   return strcmp( string, str.string ) == 0;
}

bool String::operator!=( const String& str ) const
{
   return ! operator==( str );
}


/****
 * Operators for single characters
 */
String& String::operator=( char str )
{
   string[ 0 ] = str;
   string[ 1 ] = 0;
   return *this;
}

String& String::operator+=( const char str )
{
   const int len1 = strlen( string );
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

   return *this;
}

String String::operator+( char str ) const
{
   return String( *this ) += str;
}

bool String::operator==( char str ) const
{
   return *this == String( str );
}

bool String::operator!=( char str ) const
{
   return ! operator==( str );
}


String::operator bool () const
{
   if( string[ 0 ] ) return true;
   return false;
}

bool String::operator!() const
{
   return ! operator bool();
}

String String::replace( const String& pattern,
                        const String& replaceWith,
                        int count ) const
{
   const int length = this->getLength();
   const int patternLength = pattern.getLength();
   const int replaceWithLength = replaceWith.getLength();

   int patternPointer = 0;
   int occurrences = 0;
   for( int i = 0; i < length; i++ )
   {
      if( this->string[ i ] == pattern[ patternPointer ] )
         patternPointer++;
      if( patternPointer == patternLength )
      {
         occurrences++;
         patternPointer = 0;
      }
   }
   if( count > 0 && occurrences > count )
      occurrences = count;

   String newString;
   const int newStringLength = length + occurrences * ( replaceWithLength - patternLength );
   newString.setSize( newStringLength );

   int newStringHead = 0;
   patternPointer = 0;
   occurrences = 0;
   for( int i = 0; i < length; i++ ) {
      // copy current character
      newString[ newStringHead++ ] = this->string[ i ];

      // check if pattern matches
      if( this->string[ i ] == pattern[ patternPointer ] )
         patternPointer++;
      else
         patternPointer = 0;

      // handle full match
      if( patternPointer == patternLength ) {
         // skip unwanted replacements
         if( count == 0 || occurrences < count ) {
            newStringHead -= patternLength;
            for( int j = 0; j < replaceWithLength; j++ )
               newString[ newStringHead++ ] = replaceWith[ j ];
         }
         occurrences++;
         patternPointer = 0;
      }
   }

   newString[ newStringHead ] = 0;

   return newString;
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

int String::split( Containers::List< String >& list,
                   const char separator,
                   bool skipEmpty ) const
{
   list.reset();
   String s;
   for( int i = 0; i < this->getLength(); i ++ )
   {
      if( ( *this )[ i ] == separator )
      {
         if( ! skipEmpty || s != "" )
            list.Append( s );
         s = "";
      }
      else s += ( *this )[ i ];
   }
   if( ! skipEmpty || s != "" )
      list.Append( s );
   return list.getSize();
}


bool String::save( File& file ) const
{
   TNL_ASSERT( string,
              std::cerr << "string = " << string );

   int len = strlen( string );
   if( ! file.write( &len ) )
      return false;
   if( ! file.write( string, len ) )
      return false;
   return true;
}

bool String::load( File& file )
{
   int _length;
   if( ! file.read( &_length ) ) {
      std::cerr << "I was not able to read String length." << std::endl;
      return false;
   }
   setSize( _length );
   if( _length && ! file.read( string, _length ) ) {
      std::cerr << "I was not able to read a String with a length " << length << "." << std::endl;
      return false;
   }
   string[ _length ] = 0;
   return true;
}

/*void String :: MPIBcast( int root, MPI_Comm comm )
{
#ifdef USE_MPI
   dbgFunctionName( "mString", "MPIBcast" );
   int iproc;
   MPI_Comm_rank( MPI_COMM_WORLD, &iproc );
   TNL_ASSERT( string, );
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
   std::string str;
   getline( stream, str );
   this->setString( str.c_str() );
   if( ! ( *this ) ) return false;
   return true;
}

String operator+( char string1, const String& string2 )
{
   return String( string1 ) + string2;
}

String operator+( const char* string1, const String& string2 )
{
   return String( string1 ) + string2;
}

std::ostream& operator<<( std::ostream& stream, const String& str )
{
   stream << str.getString();
   return stream;
}

} // namespace TNL
