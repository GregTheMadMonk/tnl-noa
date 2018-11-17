/***************************************************************************
                          String.h  -  description
                             -------------------
    begin                : 2004/04/10 16:35
    copyright            : (C) 2004 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <string>

namespace TNL {

class File;
class String;

/////
/// \brief Class for managing strings.
///
/// \par Example
/// \include StringExample.cpp
/// \par Output
/// \include StringExample.out
class String
: public std::string
{
public:
   /////
   /// \brief Default constructor.
   ///
   /// Constructs an empty string object.
   String() = default;

   /// \brief Default copy constructor.
   String( const String& ) = default;

   /// \brief Default move constructor.
   String( String&& ) = default;

   /// \brief Initialization by \e std::string.
   String( const std::string& str ) : std::string( str ) {}

   /// \brief Default copy assignment operator.
   String& operator=( const String& ) = default;

   /// \brief Default move assignment operator.
   String& operator=( String&& ) = default;

   /// \brief Inherited constructors and assignment operators.
   using std::string::string;
   using std::string::operator=;


   /// \brief Returns type of string - String.
   static String getType();

   /// \brief Returns the number of characters in given string. Equivalent to getSize().
   int getLength() const;

   /// \brief Returns the number of characters in given string.
   ///
   /// \par Example
   /// \include StringExampleGetSize.cpp
   /// \par Output
   /// \include StringExampleGetSize.out
   int getSize() const;

   /// \brief Returns size of allocated storage for given string.
   ///
   /// \par Example
   /// \include StringExampleGetAllocatedSize.cpp
   /// \par Output
   /// \include StringExampleGetAllocatedSize.out
   int getAllocatedSize() const;

   /////
   /// \brief Reserves space for given \e size.
   ///
   /// Requests to allocate storage space of given \e size to avoid memory reallocation.
   /// It allocates one more byte for the terminating 0.
   /// @param size Number of characters.
   ///
   /// \par Example
   /// \include StringExampleSetSize.cpp
   /// \par Output
   /// \include StringExampleSetSize.out
   void setSize( int size );

   /////
   /// \brief Returns pointer to data.
   ///
   /// It returns the content of the given string as a constant pointer to char.
   const char* getString() const;

   /////
   /// \brief Operator for accessing particular chars of the string.
   ///
   /// This function overloads operator[](). It returns a reference to
   /// the character at position \e i in given string.
   /// The character can not be changed be user.
   const char& operator[]( int i ) const;

   /// \brief Operator for accessing particular chars of the string.
   ///
   /// It returns the character at the position \e i in given string as
   /// a modifiable reference.
   char& operator[]( int i );

   /////
   // Operators for single characters.

   /// \brief This function overloads operator+=().
   ///
   /// Appends character \e str to this string.
   String& operator+=( char str );
   /// \brief This function concatenates strings and returns a newly constructed string object.
   String operator+( char str ) const;
   /// \brief This function checks whether the given string is equal to \e str.
   ///
   /// It returns \e true when the given string is equal to \e str. Otherwise returns \e false.
   bool operator==( char str ) const;
   /// \brief This function overloads operator!=().
   bool operator!=( char str ) const;

   /////
   // Operators for C strings.

   /// \brief This function overloads operator+=().
   ///
   /// It appends the C string \e str to this string.
   String& operator+=( const char* str );
   /// \brief This function concatenates C strings \e str and returns a newly
   /// constructed string object.
   String operator+( const char* str ) const;
   /// \brief This function overloads operator==().
   bool operator==( const char* str ) const;
   /// \brief This function overloads operator!=().
   bool operator!=( const char* str ) const;

   /////
   // Operators for std::string.

   /// \brief This function overloads operator+=().
   ///
   /// It appends the C string \e str to this string.
   String& operator+=( const std::string& str );
   /// \brief This function concatenates C strings \e str and returns a newly
   /// constructed string object.
   String operator+( const std::string& str ) const;
   /// \brief This function overloads operator==().
   bool operator==( const std::string& str ) const;
   /// \brief This function overloads operator!=().
   bool operator!=( const std::string& str ) const;

   /////
   // Operators for String.

   /// \brief This function overloads operator+=().
   ///
   /// It appends the C string \e str to this string.
   String& operator+=( const String& str );
   /// \brief This function concatenates C strings \e str and returns a newly
   /// constructed string object.
   String operator+( const String& str ) const;
   /// \brief This function overloads operator==().
   bool operator==( const String& str ) const;
   /// \brief This function overloads operator!=().
   bool operator!=( const String& str ) const;

   /// \brief Cast to bool operator.
   ///
   /// This function overloads operator bool(). It converts string to boolean
   /// expression (true or false).
   operator bool() const;

   /// \brief Cast to bool with negation operator.
   ///
   /// This function overloads operator!(). It converts string to boolean
   /// expression (false or true).
   bool operator!() const;

   /////
   /// \brief Replaces portion of string.
   ///
   /// Replaces section \e pattern from this string with new piece of string \e replaceWith.
   /// If parameter \e count is defined, the function makes replacement several times,
   /// every time when it finds the same pattern in this string.
   /// @param pattern Section of given string that will be replaced.
   /// @param replaceWith New piece of string that will be used to replace \e pattern.
   /// @param count A whole number that specifies how many replacements should be done.
   String replace( const String& pattern,
                   const String& replaceWith,
                   int count = 0 ) const;

   /////
   /// \brief Trims/strips given string.
   ///
   /// Removes all spaces from given string except for single spaces between words.
   String strip( char strip = ' ' ) const;

   /// \brief Splits string into list of strings with respect to given \e separator.
   ///
   /// Splits the string into list of substrings wherever \e separator occurs,
   /// and returs list of those strings. When \e separator does not appear
   /// anywhere in the given string, this function returns a single-element list
   /// containing given sting.
   /// @param separator Character, which separates substrings in given string.
   std::vector< String > split( const char separator = ' ', bool skipEmpty = false ) const;

   /////
   /// \brief Function for saving file.
   ///
   /// Writes the string to a binary file and returns boolean expression based on the
   /// success in writing into the file.
   bool save( File& file ) const;

   /////
   /// \brief Function for loading from file.
   ///
   /// Reads a string from binary file and returns boolean expression based on the
   /// success in reading the file.
   bool load( File& file );

   //! Broadcast to other nodes in MPI cluster
//   void MPIBcast( int root, MPI_Comm mpi_comm = MPI_COMM_WORLD );
};

/// \brief Returns concatenation of \e string1 and \e string2.
String operator+( char string1, const String& string2 );

/// \brief Returns concatenation of \e string1 and \e string2.
String operator+( const char* string1, const String& string2 );

/// \brief Returns concatenation of \e string1 and \e string2.
String operator+( const std::string& string1, const String& string2 );

/// \brief Performs the string output to a stream
std::ostream& operator<<( std::ostream& stream, const String& str );

template< typename T >
String convertToString( const T& value )
{
   std::stringstream str;
   str << value;
   return String( str.str().data() );
}

template<> inline String convertToString( const bool& b )
{
   if( b ) return "true";
   return "false";
}

} // namespace TNL
