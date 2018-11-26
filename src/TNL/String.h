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

namespace TNL {

class File;
namespace Containers {
   template< class T > class List;
}

class String;

template< typename T >
String convertToString( const T& value );

/////
/// \brief Class for managing strings.
///
/// \par Example
/// \include StringExample.cpp
/// \par Output
/// \include StringExample.out
class String
{
   public:

      /////
      /// \brief Basic constructor.
      ///
      /// Constructs an empty string object.
      String();

      /////
      /// \brief Constructor with char pointer.
      ///
      /// Constructs a string initialized with the 8-bit string \e c, excluding
      /// the given number of \e prefix_cut_off and \e sufix_cut_off characters.
      ///
      /// @param c Pointer to an array of characters.
      /// @param prefix_cut_off Determines the length of the prefix that is going
      /// to be omitted from the string \e c.
      /// @param sufix_cut_off Determines the length of the sufix that is going
      /// to be omitted from the string \e c.
      String( const char* c,
              int prefix_cut_off = 0,
              int sufix_cut_off = 0 );

      /// Odstranit???
      String( char* c,
              int prefix_cut_off = 0,
              int sufix_cut_off = 0 );

      /// \brief Returns type of string - String.
      static String getType();

      /////
      /// \brief Copy constructor.
      ///
      /// Constructs a copy of the string \e str.
      /// @param str Another string object, whose value is copied.
      String( const String& str );

      /// \brief Converts anything to a string.
      ///
      /// This function converts any type of value into type string.
      /// @tparam T is a type of a value to be converted
      /// @param value Word of any type (e.g. int, bool, double,...).
      template< typename T >
      String( T value )
         : string( nullptr ), length( 0 )
      {
         setString( convertToString( value ).getString() );
      }

      /// \brief Destructor.
      ~String();

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
      /// \brief Sets string from given char pointer \e c.
      ///
      /// @param prefix_cut_off determines the length of the prefix that is
      /// going to be omitted from the string \e c
      /// @param sufix_cut_off determines the length of the sufix that is going
      /// to be omitted from the string \e c
      void setString( const char* c,
                      int prefix_cut_off = 0,
                      int sufix_cut_off = 0 );

      /////
      /// \brief Returns pointer to data.
      ///
      /// It returns the content of the given string as a constant pointer to char.
      const char* getString() const;

      /// \brief Returns pointer to data.
      ///
      /// It returns the content of the given string as a non-constant pointer to char.
      char* getString();

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
      // Operators for C strings.

      /// \brief This function overloads operator=().
      ///
      /// It assigns C string \e str to this string, replacing its current contents.
      String& operator=( const char* str );
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
      // Operators for Strings.

      /// \brief This function assigns \e str to this string and returns a reference to
      /// this string.
      String& operator=( const String& str );
      /// \brief This function appends the string \e str onto the end of this string
      /// and returns a reference to this string.
      String& operator+=( const String& str );
      /// \brief This function concatenates strings \e str and returns a newly
      /// constructed string object.
      String operator+( const String& str ) const;
      /// \brief This function overloads operator==().
      ///
      /// Returns \c true if this string is equal to \e str, otherwise returns
      /// \c false.
      bool operator==( const String& str ) const;
      /// \brief This function overloads operator!=().
      ///
      /// Returns \c true if this string is not equal to \e str, otherwise
      /// returns \c false.
      bool operator!=( const String& str ) const;

      /////
      // Operators for single characters.

      /// \brief This function overloads operator=().
      ///
      /// Assigns character \e str to this string.
      String& operator=( char str );
      /// \brief This function overloads operator+=().
      ///
      /// Appends character \e str to this string.
      String& operator+=( char str );
      // \brief This function concatenates strings and returns a newly constructed string object.
      String operator+( char str ) const;
      // \brief This function concatenates strings and returns a newly constructed string object.
      bool operator==( char str ) const;
      /// \brief This function overloads operator!=().
      bool operator!=( char str ) const;

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
      /// @param list Name of list.
      /// @param separator Character, which separates substrings in given string.
      /// Empty character can not be used.
      int split( Containers::List< String >& list, const char separator = ' ' ) const;

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

      /////
      /// \brief Function for getting a line from stream.
      ///
      /// Reads one line from given stream and returns either the line or boolean
      /// expression based on the success in reading the line.
      bool getLine( std::istream& stream );

      friend std::ostream& operator<<( std::ostream& stream, const String& str );

   protected:
      
      /// \brief Pointer to char ended with zero byte.
      char* string;

      /// \brief Length of allocated piece of memory.
      int length;

}; // class String

/// \brief Returns concatenation of \e string1 and \e string2.
String operator+( char string1, const String& string2 );

/// \brief Returns concatenation of \e string1 and \e string2.
String operator+( const char* string1, const String& string2 );

/// \brief Performs the string output to a stream
std::ostream& operator<<( std::ostream& stream, const String& str );

template< typename T >
String convertToString( const T& value )
{
   std::stringstream str;
   str << value;
   return String( str.str().data() );
};

template<> inline String convertToString( const bool& b )
{
   if( b ) return "true";
   return "false";
}

} // namespace TNL
