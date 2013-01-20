/***************************************************************************
                          tnlString.h  -  description
                             -------------------
    begin                : 2004/04/10 16:35
    copyright            : (C) 2004 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef __MSTRING_H__
#define __MSTRING_H__

#include <stdio.h>
#include <iostream>
#include "mpi-supp.h"

using namespace :: std;

template< class T > class tnlList;
class tnlFile;

//! Class for managing strings
class tnlString
{
   //! Pointer to char ended with zero
   char* string;

   //! Length of the allocated piece of memory
   int length;

   public:

   //! Basic constructor
   tnlString();

   //! Constructor with char pointer
   /*! @param prefix_cut_off says length of the prefix that is going to be omitted and
       @param sufix_cut_off says the same about sufix.
    */
   tnlString( const char* c,
              int prefix_cut_off = 0,
              int sufix_cut_off = 0 );

   //! Copy constructor
   tnlString( const tnlString& str );

   //! Convert number to a string
   tnlString( int number );
   
   tnlString( long int number );

   tnlString( float number );

   tnlString( double number );

   //! Destructor
   ~tnlString();

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
    * for example tnlString + const char*
    */

   //! Operator =
   tnlString& operator = ( const tnlString& str );

   //! Operator +=
   tnlString& operator += ( const char* str );

   //! Operator +=
   tnlString& operator += ( const char str );

   //! Operator +=
   tnlString& operator += ( const tnlString& str );
 
   //! Operator +
   tnlString operator + ( const tnlString& str );

   //! Operator +
   tnlString operator + ( const char* str );

   //! Comparison operator 
   bool operator == ( const tnlString& str ) const;

   //! Comparison operator 
   bool operator != ( const tnlString& str ) const;

   //! Comparison operator
   bool operator == ( const char* ) const;

   //! Comparison operator
   bool operator != ( const char* ) const;
  
   //! Retyping operator
   operator bool () const;

   //! Return length of the string
   int getLength() const;

   // TODO: remove
   //! Write to a binary file
   bool save( ostream& file ) const;

   // TODO: remove
   //! Read from binary file
   bool load( istream& file );

   //! Write to a binary file
   bool save( tnlFile& file ) const;

   //! Read from binary file
   bool load( tnlFile& file );

   //! Broadcast to other nodes in MPI cluster
   void MPIBcast( int root, MPI_Comm mpi_comm = MPI_COMM_WORLD );

   //! Read one line from given stream.
   bool getLine( istream& stream );

   //! Parse the string into list of strings w.r.t. given separator.
   int parse( tnlList< tnlString >& list, const char separator = ' ' ) const;

   friend ostream& operator << ( ostream& stream, const tnlString& str );
};

inline tnlString GetParameterType( const tnlString& ) { return tnlString( "tnlString" ); };

ostream& operator << ( ostream& stream, const tnlString& str );

#endif
