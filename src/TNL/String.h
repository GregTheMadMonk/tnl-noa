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

//! Class for managing strings
class String
{
   //! Pointer to char ended with zero
   char* string;

   //! Length of the allocated piece of memory
   int length;

public:
   //! Basic constructor
   String();
   
   //! Constructor from const char*
   String( const char* str );   

   //! Constructor with char pointer
   /*! @param prefix_cut_off says length of the prefix that is going to be omitted and
       @param sufix_cut_off says the same about sufix.
    */
   String( const char* c,
           int prefix_cut_off,
           int sufix_cut_off = 0 );

   //! Copy constructor
   String( const String& str );
   
   //////
   /// Templated constructor
   ///
   /// It must be explicit otherwise it is called recursively from inside of its
   /// definition ( in operator << ). It leads to stack overflow and segmentation fault.
   template< typename T >
   explicit
   String( T value )
      : string( nullptr ), length( 0 )
   {
      std::stringstream str;
      str << value;
      setString( str.str().data() );
   }

   //! Destructor
   ~String();
   
   static String getType();

   //! Return length of the string
   int getLength() const;
   int getSize() const;

   //! Return currently allocated size
   int getAllocatedSize() const;

   //! Reserve space for given number of characters
   void setSize( int size );

   //! Set string from given char pointer
   /*! @param prefix_cut_off says length of the prefix that is going to be omitted and
       @param sufix_cut_off says the same about sufix.
    */
   void setString( const char* c,
                   int prefix_cut_off = 0,
                   int sufix_cut_off = 0 );

   //! Return pointer to data
   const char* getString() const;

   //! Return pointer to data
   char* getString();

   //! Operator for accesing particular chars of the string
   const char& operator[]( int i ) const;

   //! Operator for accesing particular chars of the string
   char& operator[]( int i );

   /****
    * Operators for C strings
    */
   String& operator=( const char* str );
   String& operator+=( const char* str );
   String operator+( const char* str ) const;
   bool operator==( const char* str ) const;
   bool operator!=( const char* str ) const;
 
   /****
    * Operators for Strings
    */
   String& operator=( const String& str );
   String& operator+=( const String& str );
   String operator+( const String& str ) const;
   bool operator==( const String& str ) const;
   bool operator!=( const String& str ) const;

   /****
    * Operators for single characters
    */
   String& operator=( char str );
   String& operator+=( char str );
   String operator+( char str ) const;
   bool operator==( char str ) const;
   bool operator!=( char str ) const;

   //! Cast to bool operator
   operator bool() const;

   //! Cast to bool with negation operator
   bool operator!() const;

   String replace( const String& pattern,
                   const String& replaceWith,
                   int count = 0 ) const;

   String strip( char strip = ' ' ) const;

   //! Split the string into list of strings w.r.t. given separator.
   int split( Containers::List< String >& list, const char separator = ' ' ) const;

   //! Write to a binary file
   bool save( File& file ) const;

   //! Read from binary file
   bool load( File& file );

   //! Broadcast to other nodes in MPI cluster
//   void MPIBcast( int root, MPI_Comm mpi_comm = MPI_COMM_WORLD );

   //! Read one line from given stream.
   bool getLine( std::istream& stream );

   friend std::ostream& operator<<( std::ostream& stream, const String& str );
};

String operator+( char string1, const String& string2 );

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
