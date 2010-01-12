/***************************************************************************
                          tnlMatrix.h  -  description
                             -------------------
    begin                : 2007/07/23
    copyright            : (C) 2007 by Tomas Oberhuber
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

#ifndef tnlMatrixH
#define tnlMatrixH

#include <ostream>
#include <iomanip.h>
#include <core/tnlObject.h>

template< typename T > class tnlMatrix : public tnlObject
{
   public:

   virtual int GetSize() const = 0;

   virtual T GetElement( int row, int column ) const = 0;

   //! Setting given element
   /*! Returns false if fails to allocate the new element
    */
   virtual bool SetElement( int row, int column, const T& v ) = 0;

   virtual bool AddToElement( int row, int column, const T& v ) = 0;
   
   virtual T RowProduct( const int row, const T* vec ) const = 0;
   
   virtual void VectorProduct( const T* vec, T* result ) const = 0;

   virtual T GetRowL1Norm( int row ) const = 0;

   virtual void MultiplyRow( int row, const T& value ) = 0;

   virtual ~tnlMatrix()
   {};
};

//! Operator <<
template< typename T > ostream& operator << ( ostream& o_str, const tnlMatrix< T >& A )
{
   int size = A. GetSize();
   int i, j;
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
