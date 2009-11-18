/***************************************************************************
                          mFullMatrix.h  -  description
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

#ifndef mFullMatrixH
#define mFullMatrixH

#include <core/mField2D.h>
#include <matrix/mBaseMatrix.h>


template< typename T > class mFullMatrix : public mBaseMatrix< T >, public mField2D< T >
{

   public:

   //! Basic constructor
   mFullMatrix(){};

   //! Constructor with matrix dimension
   mFullMatrix( const long int size )
   : mField2D< T >( size, size ){};

   tnlString GetType() const
   {
      T t;
      return tnlString( "mFullMatrix< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
   };

   const tnlString& GetMatrixClass() const
   {
      return mMatrixClass :: main;
   };

   long int GetSize() const
   {
      return mField2D< T > :: GetXSize(); // it is the same as GetYSize()
   };
   
   T GetElement( long int i, long int j ) const
   {
      return ( *this )( i, j );
   };

   bool SetElement( long int i, long int j, const T& v )
   {
      ( *this )( i, j ) = v;
      return true;
   };

   bool AddToElement( long int i, long int j, const T& v )
   {
      ( *this )( i, j ) += v;
      return true;
   };

   //! Row product
   /*! Compute product of given vector with given row
    */
   T RowProduct( const long int row, const T* vec ) const
   {
      const long int size = GetSize();
      long int pos = row * size;
      const T* data = mField2D< T > :: Data();
      T res( 0.0 );
      long int i;
      for( i = 0; i < size; i ++ )
      {
         res += data[ pos ] * vec[ i ];
         pos ++; 
      }
      return res;
   };

   //! Vector product
   void VectorProduct( const T* vec, T* result ) const
   {
      const long int size = GetSize();
      long int pos( 0 );
      const T* data = mField2D< T > :: Data();
      T res;
      long int i, j;
      for( i = 0; i < size; i ++ )
      {
         res = 0.0;
         for( j = 0; j < size; j ++ )
         {
            res += data[ pos ] * vec[ j ];
            pos ++; 
         }
         result[ i ] = res;
      }
   };


   //! Multiply row
   void MultiplyRow( const long int row, const T& c )
   {
      const long int size = GetSize();
      T* data = mField2D< T > :: Data();
      long int i;
      long int pos = row * size;
      for( i = 0; i < size; i ++ )
      {
         data[ pos + i ] *= c;
      }
   };

   //! Get row L1 norm
   T GetRowL1Norm( const long int row ) const
   {
      const long int size = GetSize();
      const T* data = mField2D< T > :: Data();
      T res( 0.0 );
      long int i;
      long int pos = row * size;
      for( i = 0; i < size; i ++ )
         res += fabs( data[ pos + i ] );
      return res;
   };

   //! Destructor
   ~mFullMatrix(){};
};

//! Matrix product
template< typename T > void MatrixProduct( const mFullMatrix< T >& m1,
                                           const mFullMatrix< T >& m2,
                                           mFullMatrix< T >& result )
{
   assert( m1. GetSize() == m2. GetSize() && m2. GetSize() == result. GetSize() );
   long int size = result. GetSize();
   long int i, j, k;
   for( i = 0; i < size; i ++ )
      for( j = 0; j < size; j ++ )
      {
         T res( 0.0 );
         for( k = 0; k < size; k ++ )
            res += m1( i, k ) * m2( k, j ); 
         result( i, j ) = res;
      }
};

//! Matrix sum
template< typename T > void MatrixSum( const mFullMatrix< T >& m1,
                                       const mFullMatrix< T >& m2,
                                       mFullMatrix< T >& result )
{
   assert( m1. GetSize() == m2. GetSize() && m2. GetSize() == result. GetSize() );
   long int size = result. GetSize();
   long int i,j;
   for( i = 0; i < size; i ++ )
      for( j = 0; j < size; j ++ )
         result( i, j ) = m1( i, j ) + m2( i, j );
};

//! Operator <<
template< typename T > ostream& operator << ( ostream& o_str, const mFullMatrix< T >& A )
{
   return operator << ( o_str, ( const mBaseMatrix< T >& ) A );
};

#endif
