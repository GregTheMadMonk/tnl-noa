/***************************************************************************
                          String.h  -  description
                             -------------------
    begin                : 2004/04/10 16:35
    copyright            : (C) 2004 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <TNL/mpi-supp.h>


namespace TNL {

class File;
namespace Containers {
   template< class T > class List;
}

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

   //! Constructor with char pointer
   /*! @param prefix_cut_off says length of the prefix that is going to be omitted and
       @param sufix_cut_off says the same about sufix.
    */
   String( const char* c,
              int prefix_cut_off = 0,
              int sufix_cut_off = 0 );

   static String getType();

   //! Copy constructor
   String( const String& str );

   //! Convert number to a string
   String( unsigned number );

   String( int number );
 
   String( long int number );

   String( float number );

   String( double number );

   //! Destructor
   ~String();

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
    * TODO: the operators do not work properly
    * for example String + const char*
    */

   //! Operator =
   String& operator = ( const String& str );

   //! Operator +=
   String& operator += ( const char* str );

   //! Operator +=
   String& operator += ( const char str );

   //! Operator +=
   String& operator += ( const String& str );
 
   //! Operator +
   String operator + ( const String& str ) const;

   //! Operator +
   String operator + ( const char* str ) const;

   //! Comparison operator
   bool operator == ( const String& str ) const;

   //! Comparison operator
   bool operator != ( const String& str ) const;

   //! Comparison operator
   bool operator == ( const char* ) const;

   //! Comparison operator
   bool operator != ( const char* ) const;
 
   //! Retyping operator
   operator bool () const;

   //! Return length of the string
   int getLength() const;

   void replace( const String& pattern,
                 const String& replaceWith );

   String strip( char strip = ' ' ) const;

   // TODO: remove
   //! Write to a binary file
   bool save( std::ostream& file ) const;

   // TODO: remove
   //! Read from binary file
   bool load( std::istream& file );

   //! Write to a binary file
   bool save( File& file ) const;

   //! Read from binary file
   bool load( File& file );

   //! Broadcast to other nodes in MPI cluster
//   void MPIBcast( int root, MPI_Comm mpi_comm = MPI_COMM_WORLD );

   //! Read one line from given stream.
   bool getLine( std::istream& stream );

   //! Parse the string into list of strings w.r.t. given separator.
   int parse( Containers::List< String >& list, const char separator = ' ' ) const;

   friend std::ostream& operator << ( std::ostream& stream, const String& str );
};

String operator + ( const char* string1, const String& string2 );

std::ostream& operator << ( std::ostream& stream, const String& str );

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
