/***************************************************************************
                          mString.h  -  description
                             -------------------
    begin                : 2004/04/10 16:35
    copyright            : (C) 2004 by Tomas Oberhuber
    email                : tomasoberhuber@seznam.cz
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


//! Class for managing strings
class mString
{
   //! Pointer to char ended with zero
   char* string;

   //! Length of the allocated piece of memory
   int length;

   public:

   //! Basic constructor
   mString();

   //! Constructor with char pointer
   /*! @param prefix_cut_off says length of the prefix that is going to be omitted and
       @param sufix_cut_off says the same about sufix.
    */
   mString( const char* c,
            int prefix_cut_off = 0,
            int sufix_cut_off = 0 );

   //! Copy constructor
   mString( const mString& str );

   //! Destructor
   ~mString();

   //! Set string from given char pointer
   /*! @param prefix_cut_off says length of the prefix that is going to be omitted and
       @param sufix_cut_off says the same about sufix.
    */
   void SetString( const char* c,
                   int prefix_cut_off = 0,
                   int sufix_cut_off = 0 );

   //! Operator =
   mString& operator = ( const mString& str );

   //! Operator +=
   mString& operator += ( const char* str );

   //! Operator +=
   mString& operator += ( const mString& str );
 
   //! Operator +
   mString operator + ( const mString& str );

   //! Comparison operator 
   bool operator == ( const mString& str ) const;

   //! Comparison operator 
   bool operator != ( const mString& str ) const;

   //! Comparison operator
   bool operator == ( const char* ) const;

   //! Comparison operator
   bool operator != ( const char* ) const;
  
   //! Retyping operator
   operator bool () const;

   //! Return length of the string
   int Length() const;

   //! Return pointer to data
   const char* Data() const;

   //! Write to a binary file
   bool Save( ostream& file ) const;

   //! Read from binary file
   bool Load( istream& file );

   //! Broadcast to other nodes in MPI cluster
   void MPIBcast( int root, MPI_Comm mpi_comm = MPI_COMM_WORLD );

   friend ostream& operator << ( ostream& stream, const mString& str );
};

inline mString GetParameterType( const mString& ) { return mString( "mString" ); };

ostream& operator << ( ostream& stream, const mString& str );

#endif
