/***************************************************************************
                          mField2D.h  -  description
                             -------------------
    begin                : 2005/08/10
    copyright            : (C) 2005 by Tomas Oberhuber
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

#ifndef mField2DH
#define mField2DH

#include <string.h>
#include "tnlObject.h"
#include "mLongVector.h"

template< class T > class mField2D : public mLongVector< T >
{
   public:

   mField2D()
   : mLongVector< T >( 0 )
   { };

   mField2D( long int _x_size,
             long int _y_size )
   : mLongVector< T >( _x_size * _y_size ),
     x_size( _x_size ), y_size( _y_size )
   { };

   mField2D( const mField2D& f )
   : mLongVector< T >( f ),
     x_size( f. x_size ), y_size( f. y_size )
   { };

   tnlString GetType() const
   {
      T t;
      return tnlString( "mField2D< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
   };

   long int GetXSize() const
   {
      return x_size;
   };

   long int GetYSize() const
   {
      return y_size;
   };

   bool SetNewDimensions( long int new_x_size,
                          long int new_y_size )
   {
      x_size = new_x_size;
      y_size = new_y_size;
      return mLongVector< T > :: SetNewSize( x_size * y_size );
   };

   bool SetNewDimensions( const mField2D< T >& f )
   {
      return SetNewDimensions( f. GetXSize(), f. GetYSize() );
   };

   void SetSharedData( T* _data, const long int _x_size, const long int _y_size )
   {
      mLongVector< T > :: SetSharedData( _data, _x_size * _y_size );
      x_size = _x_size;
      y_size = _y_size;
   };

   const T& operator() ( long int i, long int j ) const
   {
      assert( i < x_size && j < y_size && i >= 0 && j >= 0 );
      return mLongVector< T > :: data[ i * y_size + j ];
   };

   T& operator() ( long int i, long int j )
   {
      assert( i < x_size && j < y_size && i >= 0 && j >= 0 );
      return mLongVector< T > :: data[ i * y_size + j ];
   };

   long int GetLongVectorIndex( long int i, long int j ) const
   {
      assert( i >= 0 && j >= 0 );
      assert( i < x_size && j < y_size );
      return i * y_size + j;
   };
   
   //! Method for saving the object to a file as a binary data
   bool Save( ostream& file ) const
   {
      if( ! mLongVector< T > :: Save( file ) ) return false;
      file. write( ( char* ) &x_size, sizeof( long int ) );
      file. write( ( char* ) &y_size, sizeof( long int ) );
      if( file. bad() ) return false;
      return true;
   };

   //! Method for restoring the object from a file
   bool Load( istream& file )
   {
      if( ! mLongVector< T > :: Load( file ) ) return false;
      file. read( ( char* ) &x_size, sizeof( long int ) );
      file. read( ( char* ) &y_size, sizeof( long int ) );
      if( file. bad() ) return false;
      return true;
   };   
   
   protected:

   long int x_size, y_size;
};

// Explicit instatiation
template class mField2D< double >;

#endif
