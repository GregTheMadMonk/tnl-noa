/***************************************************************************
                          String.cpp  -  description
                             -------------------
    begin                : 2004/04/10 16:36
    copyright            : (C) 2004 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/String.h>
#include <TNL/Assert.h>
#include <TNL/File.h>
#include <TNL/Math.h>
//#ifdef USE_MPI
//   #include <mpi.h>
//#endif

namespace TNL {

String String::getType()
{
   return String( "String" );
}

int String::getLength() const
{
   return getSize();
}

int String::getSize() const
{
   return this->size();
}

int String::getAllocatedSize() const
{
   return this->capacity();
}

void String::setSize( int size )
{
   TNL_ASSERT_GE( size, 0, "string size must be non-negative" );
   this->reserve( size );
}

const char* String::getString() const
{
   return this->c_str();
}


const char& String::operator[]( int i ) const
{
   TNL_ASSERT( i >= 0 && i < getLength(),
               std::cerr << "Accessing char outside the string." );
   return std::string::operator[]( i );
}

char& String::operator[]( int i )
{
   TNL_ASSERT( i >= 0 && i < getLength(),
               std::cerr << "Accessing char outside the string." );
   return std::string::operator[]( i );
}


/****
 * Operators for single characters
 */
String& String::operator+=( char str )
{
   std::string::operator+=( str );
   return *this;
}

String String::operator+( char str ) const
{
   return String( *this ) += str;
}

bool String::operator==( char str ) const
{
   return std::string( *this ) == std::string( 1, str );
}

bool String::operator!=( char str ) const
{
   return ! operator==( str );
}


/****
 * Operators for C strings
 */
String& String::operator+=( const char* str )
{
   std::string::operator+=( str );
   return *this;
}

String String::operator+( const char* str ) const
{
   return String( *this ) += str;
}

bool String::operator==( const char* str ) const
{
   return std::string( *this ) == str;
}

bool String::operator!=( const char* str ) const
{
   return ! operator==( str );
}


/****
 * Operators for std::string
 */
String& String::operator+=( const std::string& str )
{
   std::string::operator+=( str );
   return *this;
}

String String::operator+( const std::string& str ) const
{
   return String( *this ) += str;
}

bool String::operator==( const std::string& str ) const
{
   return std::string( *this ) == str;
}

bool String::operator!=( const std::string& str ) const
{
   return ! operator==( str );
}


/****
 * Operators for String
 */
String& String::operator+=( const String& str )
{
   std::string::operator+=( str );
   return *this;
}

String String::operator+( const String& str ) const
{
   return String( *this ) += str;
}

bool String::operator==( const String& str ) const
{
   return std::string( *this ) == str;
}

bool String::operator!=( const String& str ) const
{
   return ! operator==( str );
}


String::operator bool () const
{
   return getLength();
}

bool String::operator!() const
{
   return ! operator bool();
}

String String::replace( const String& pattern,
                        const String& replaceWith,
                        int count ) const
{
   std::string newString = *this;

   std::size_t index = 0;
   for( int i = 0; i < count || count == 0; i++ ) {
      // locate the substring to replace
      index = newString.find( pattern, index );
      if( index == std::string::npos )
         break;

      // make the replacement
      newString.replace( index, pattern.getLength(), replaceWith );
      index += replaceWith.getLength();
   }

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
      return substr( prefix_cut_off, getLength() - prefix_cut_off - sufix_cut_off );
   return "";
}

std::vector< String > String::split( const char separator, bool skipEmpty ) const
{
   std::vector< String > parts;
   String s;
   for( int i = 0; i < this->getLength(); i++ ) {
      if( ( *this )[ i ] == separator ) {
         if( ! skipEmpty || s != "" )
            parts.push_back( s );
         s = "";
      }
      else s += ( *this )[ i ];
   }
   if( ! skipEmpty || s != "" )
      parts.push_back( s );
   return parts;
}


bool String::save( File& file ) const
{
   const int len = getLength();
   if( ! file.write( &len ) )
      return false;
   if( ! file.write( this->c_str(), len ) )
      return false;
   return true;
}

bool String::load( File& file )
{
   int length;
   if( ! file.read( &length ) ) {
      std::cerr << "I was not able to read String length." << std::endl;
      return false;
   }
   char buffer[ length ];
   if( length && ! file.read( buffer, length ) ) {
      std::cerr << "I was not able to read a String with a length " << length << "." << std::endl;
      return false;
   }
   this->assign( buffer, length );
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

String operator+( char string1, const String& string2 )
{
   return convertToString( string1 ) + string2;
}

String operator+( const char* string1, const String& string2 )
{
   return String( string1 ) + string2;
}

String operator+( const std::string& string1, const String& string2 )
{
   return String( string1 ) + string2;
}

std::ostream& operator<<( std::ostream& stream, const String& str )
{
   stream << str.getString();
   return stream;
}

} // namespace TNL
