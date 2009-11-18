/***************************************************************************
                          mBaseMatrix.h  -  description
                             -------------------
    begin                : 2007/07/23
    copyright            : (C) 2007 by Tomá¹ Oberhuber
    email                : oberhuber@seznam.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef mBaseMatrixH
#define mBaseMatrixH

#include <ostream>
#include <iomanip>
#include <core/tnlObject.h>
#include <core/tnlString.h>

class mMatrixClass
{
   private:

   mMatrixClass() {};

   public:
   static const tnlString main;
   static const tnlString petsc;
};

template< typename T > class mBaseMatrix : public tnlObject
{
   public:

   //! Matrix class tells what implementation of matrix we want.
   /*! Matrix class can be main, PETSC, CUDA etc.
    */
   virtual const tnlString& GetMatrixClass() const = 0;

   virtual long int GetSize() const = 0;

   virtual T GetElement( long int row, long int column ) const = 0;

   //! Setting given element
   /*! Returns false if fails to allocate the new element
    */
   virtual bool SetElement( long int row, long int column, const T& v ) = 0;

   virtual bool AddToElement( long int row, long int column, const T& v ) = 0;
   
   virtual T RowProduct( const long int row, const T* vec ) const = 0;
   
   virtual void VectorProduct( const T* vec, T* result ) const = 0;

   virtual T GetRowL1Norm( long int row ) const = 0;

   virtual void MultiplyRow( long int row, const T& value ) = 0;

   virtual ~mBaseMatrix()
   {};
};

//! Operator <<
template< typename T > ostream& operator << ( ostream& o_str, const mBaseMatrix< T >& A )
{
   long int size = A. GetSize();
   long int i, j;
   o_str << endl;
   for( i = 0; i < size; i ++ )
   {
      for( j = 0; j < size; j ++ )
      {
         const T& v = A. GetElement( i, j );
         if( v == 0.0 ) o_str << setw( 12 ) << ".";
         else o_str << setprecision( 6 ) << setw( 12 ) << v;
      }
      o_str << endl;
   }
   return o_str;
};

#endif
