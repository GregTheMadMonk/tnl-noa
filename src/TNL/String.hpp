/***************************************************************************
                          String_impl.h  -  description
                             -------------------
    begin                : 2004/04/10 16:36
    copyright            : (C) 2004 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/String.h>
#include <TNL/Assert.h>
#include <TNL/Math.h>
#ifdef HAVE_MPI
   #include <mpi.h>
#endif

namespace TNL {

inline String String::getType()
{
   return String( "String" );
}

inline int String::getLength() const
{
   return getSize();
}

inline int String::getSize() const
{
   return this->size();
}

inline int String::getAllocatedSize() const
{
   return this->capacity();
}

inline void String::setSize( int size )
{
   TNL_ASSERT_GE( size, 0, "string size must be non-negative" );
   this->reserve( size );
}

inline const char* String::getString() const
{
   return this->c_str();
}

inline const char& String::operator[]( int i ) const
{
   TNL_ASSERT( i >= 0 && i < getLength(),
               std::cerr << "Accessing char outside the string." );
   return std::string::operator[]( i );
}

inline char& String::operator[]( int i )
{
   TNL_ASSERT( i >= 0 && i < getLength(),
               std::cerr << "Accessing char outside the string." );
   return std::string::operator[]( i );
}

/****
 * Operators for single characters
 */
inline String& String::operator+=( char str )
{
   std::string::operator+=( str );
   return *this;
}

inline String String::operator+( char str ) const
{
   return String( *this ) += str;
}

inline bool String::operator==( char str ) const
{
   return std::string( *this ) == std::string( 1, str );
}

inline bool String::operator!=( char str ) const
{
   return ! operator==( str );
}

/****
 * Operators for C strings
 */
inline String& String::operator+=( const char* str )
{
   std::string::operator+=( str );
   return *this;
}

inline String String::operator+( const char* str ) const
{
   return String( *this ) += str;
}

inline bool String::operator==( const char* str ) const
{
   return std::string( *this ) == str;
}

inline bool String::operator!=( const char* str ) const
{
   return ! operator==( str );
}

/****
 * Operators for std::string
 */
inline String& String::operator+=( const std::string& str )
{
   std::string::operator+=( str );
   return *this;
}

inline String String::operator+( const std::string& str ) const
{
   return String( *this ) += str;
}

inline bool String::operator==( const std::string& str ) const
{
   return std::string( *this ) == str;
}

inline bool String::operator!=( const std::string& str ) const
{
   return ! operator==( str );
}

/****
 * Operators for String
 */
inline String& String::operator+=( const String& str )
{
   std::string::operator+=( str );
   return *this;
}

inline String String::operator+( const String& str ) const
{
   return String( *this ) += str;
}

inline bool String::operator==( const String& str ) const
{
   return std::string( *this ) == str;
}

inline bool String::operator!=( const String& str ) const
{
   return ! operator==( str );
}


inline String::operator bool () const
{
   return getLength();
}

inline bool String::operator!() const
{
   return ! operator bool();
}

inline String
String::replace( const String& pattern,
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

inline String
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

inline std::vector< String >
String::split( const char separator, SplitSkipEmpty skipEmpty ) const
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

inline String operator+( char string1, const String& string2 )
{
   return convertToString( string1 ) + string2;
}

inline String operator+( const char* string1, const String& string2 )
{
   return String( string1 ) + string2;
}

inline String operator+( const std::string& string1, const String& string2 )
{
   return String( string1 ) + string2;
}

inline std::ostream& operator<<( std::ostream& stream, const String& str )
{
   stream << str.getString();
   return stream;
}

#ifdef HAVE_MPI
inline void send( const String& str, int target, int tag, MPI_Comm mpi_comm )
{
   int size = str.getSize();
   MPI_Send( &size, 1, MPI_INT, target, tag, mpi_comm );
   MPI_Send( str.getString(), str.length(), MPI_CHAR, target, tag, mpi_comm );
}

inline void receive( String& str, int source, int tag, MPI_Comm mpi_comm )
{
   int size;
   MPI_Status status;
   MPI_Recv( &size, 1, MPI_INT, source, tag, mpi_comm, &status );
   str.setSize( size );
   MPI_Recv( const_cast< void* >( ( const void* ) str.data() ), size, MPI_CHAR, source, tag, mpi_comm,  &status );
}

/*
inline void String :: MPIBcast( int root, MPI_Comm comm )
{
   int iproc;
   MPI_Comm_rank( MPI_COMM_WORLD, &iproc );
   TNL_ASSERT( string, );
   int len = strlen( string );
   MPI_Bcast( &len, 1, MPI_INT, root, comm );
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
}
*/
#endif


} // namespace TNL
