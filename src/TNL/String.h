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
// \brief Class for managing strings.
class String
{
   public:

      /////
      /// \brief Basic constructor. Constructs an empty string object, with the length of zero characters.
      String();

      /////
      /// \brief Constructor with char pointer.      
      /// Copies the null-terminated character sequence (C-string) pointed by \e c.
      /// Constructs a string initialized with the 8-bit string \e c, excluding the given number of \e prefix_cut_off and \e sufix_cut_off characters.
      /// @param prefix_cut_off determines the length of the prefix that is going to be omitted from the string \e c
      /// @param sufix_cut_off determines the length of the sufix that is going to be omitted from the string \e c
      String( const char* c,
              int prefix_cut_off = 0,
              int sufix_cut_off = 0 );

      String( char* c,
              int prefix_cut_off = 0,
              int sufix_cut_off = 0 );

      /// Returns type of string - String.
      static String getType();

      /////
      /// \brief Copy constructor. Constructs a copy of the string \e str.
      String( const String& str );

      /// \brief Converts anything to a string.
      template< typename T >
      String( T value )
         : string( nullptr ), length( 0 )
      {
         setString( convertToString( value ).getString() );
      }

      /// \brief Destructor.
      ~String();

      /// Returns the number of characters in given string. Equivalent to \c getSize().
      int getLength() const;
      /// Returns the number of characters in given string.
      int getSize() const;

      /// Returns size of allocated storage for given string. 
      int getAllocatedSize() const;

      /// Reserves space for given number of characters (\e size).
      /// Requests to allocate storage for given \e size (number of characters).
      void setSize( int size );

      /////
      /// Sets string from given char pointer \e c.
      /// @param prefix_cut_off determines the length of the prefix that is going to be omitted from the string \e c
      /// @param sufix_cut_off determines the length of the sufix that is going to be omitted from the string \e c
      void setString( const char* c,
                      int prefix_cut_off = 0,
                      int sufix_cut_off = 0 );

      /// Returns pointer to data. It returns the content of the given string.
      const char* getString() const;

      /// Returns pointer to data.
      char* getString();

      /// \brief Operator for accesing particular chars of the string. This function overloads operator[](). It returns a reference to the character at position \e i in given string.
      const char& operator[]( int i ) const;

      /// \brief Operator for accesing particular chars of the string. It returns the character at the position \e i in given string as a modifiable reference.
      char& operator[]( int i );

      /////
      /// \brief Operators for C strings.
      /// This function overloads operator=(). It assigns \e str to this string, replacing its current contents.
      String& operator=( const char* str );
      /// This function overloads operator+=(). It appends the string \e str to this string.
      String& operator+=( const char* str );
      /// This function concatenates strings and returns a newly constructed string object.
      String operator+( const char* str ) const;
      /// This function overloads operator==().
      bool operator==( const char* str ) const;
      /// This function overloads operator!=().
      bool operator!=( const char* str ) const;

      /////
      /// \brief Operators for Strings.
      /// This function assigns \e str to this string and returns a reference to this string.
      String& operator=( const String& str );
      /// This function appends the string \e str onto the end of this string and returns a reference to this string.
      String& operator+=( const String& str );
      /// This function concatenates strings and returns a newly constructed string object.
      String operator+( const String& str ) const;
      /// This function overloads operator==(). It returns \c true if this string is equal to \e str, otherwise returns \c false.
      bool operator==( const String& str ) const;
      /// This function overloads operator!=(). It returns \c true if this string is not equal to \e str, otherwise returns \c false.
      bool operator!=( const String& str ) const;

      /////
      /// \brief Operators for single characters.
      /// This function overloads operator=(). It assigns character /e str to this string.
      String& operator=( char str );
      /// This function overloads operator+=(). It appends character /e str to this string.
      String& operator+=( char str );
      // This function concatenates strings and returns a newly constructed string object.
      String operator+( char str ) const;
      // This function concatenates strings and returns a newly constructed string object.
      bool operator==( char str ) const;
      /// This function overloads operator!=().
      bool operator!=( char str ) const;

      /// \brief Cast to bool operator.
      operator bool() const;

      /// \brief Cast to bool with negation operator.
      bool operator!() const;

      /////
      /// \brief Replaces portion of string.
      ///It replaces section \e pattern from this string with new piece of string \e replaceWith.
      ///If parameter \e count is defined, the function makes replacement several times, every time when it finds the same pattern in this string.
      String replace( const String& pattern,
                      const String& replaceWith,
                      int count = 0 ) const;

      /// \brief Trims/strips given string. It removes all spaces from string except for single spaces between words.
      String strip( char strip = ' ' ) const;

      /// \brief Splits string into list of strings with respect to given \e separator.
      /// It splits the string into list of substrings wherever \e separator occurs, and returs list of those strings.
      /// When \e separator does not appear anywhere in the given string, this function returns a single-element list containing given sting.
      /// @param separator Character, which separates substrings in given string. Empty character can not be used.
      int split( Containers::List< String >& list, const char separator = ' ' ) const;

      /// Write to a binary file
      bool save( File& file ) const;

      /// Read from binary file
      bool load( File& file );

      //! Broadcast to other nodes in MPI cluster
      //   void MPIBcast( int root, MPI_Comm mpi_comm = MPI_COMM_WORLD );

      /// Read one line from given stream.
      bool getLine( std::istream& stream );

      friend std::ostream& operator<<( std::ostream& stream, const String& str );

   protected:
      /// Pointer to char ended with zero
      char* string;

      /// Length of the allocated piece of memory
      int length;


};

/// Returns concatenation of \e string1 and \e string2.
String operator+( char string1, const String& string2 );

/// Returns concatenation of \e string1 and \e string2.
String operator+( const char* string1, const String& string2 );

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
