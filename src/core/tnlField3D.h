/***************************************************************************
                          tnlField3D.h  -  description
                             -------------------
    begin                : 2009/07/21
    copyright            : (C) 2009 by Tomas Oberhuber
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

#ifndef tnlField3DH
#define tnlField3DH

#include <string.h>
#include "tnlObject.h"
#include "tnlLongVector.h"

template< class T > class tnlField3D : public tnlLongVector< T >
{
   public:

   tnlField3D( const char* name = 0 )
   : tnlLongVector< T >( name )
   { };

   tnlField3D( const char* name,
               int _x_size,
               int _y_size,
               int _z_size )
   : tnlLongVector< T >( name, _x_size * _y_size * _z_size ),
     x_size( _x_size ), y_size( _y_size ), z_size( _z_size )
   { };

   tnlField3D( const tnlField3D& f )
   : tnlLongVector< T >( f ),
     x_size( f. x_size ), y_size( f. y_size ), z_size( f.z_size )
   { };

   tnlString GetType() const
   {
      T t;
      return tnlString( "tnlField3D< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
   };

   int GetXSize() const
   {
      return x_size;
   };

   int GetYSize() const
   {
      return y_size;
   };

   int GetZSize() const
   {
      return z_size;
   };

   bool SetNewDimensions( int new_x_size,
                          int new_y_size,
                          int new_z_size )
   {
      x_size = new_x_size;
      y_size = new_y_size;
      z_size = new_z_size;
      return tnlLongVector< T > :: SetNewSize( x_size * y_size * z_size );
   };

   bool SetNewDimensions( const tnlField3D< T >& f )
   {
      return SetNewDimensions( f. GetXSize(), f. GetYSize(), f. GetZSize() );
   };

   void SetSharedData( T* _data, const int _x_size, const int _y_size, const int _z_size )
   {
      tnlLongVector< T > :: SetSharedData( _data, _x_size * _y_size * _z_size );
      x_size = _x_size;
      y_size = _y_size;
      z_size = _z_size;
   };

   const T& operator() ( int i, int j, int k ) const
   {
      assert( i < x_size && j < y_size && k < z_size && i >= 0 && j >= 0 && k >= 0 );
      return tnlLongVector< T > :: data[ i * y_size * z_size + j * z_size + k ];
   };

   T& operator() ( int i, int j, int k )
   {
      assert( i < x_size && j < y_size && k < z_size && i >= 0 && j >= 0 && k >= 0 );
      return tnlLongVector< T > :: data[ i * y_size * z_size + j * z_size + k ];
   };

   int GetLongVectorIndex( int i, int j, int k ) const
   {
      assert( i >= 0 && j >= 0 && k >= 0 );
      assert( i < x_size && j < y_size && k < z_size );
      return i * y_size * z_size + j * z_size + k;
   };
   
   //! Method for saving the object to a file as a binary data
   bool Save( ostream& file ) const
   {
      if( ! tnlLongVector< T > :: Save( file ) ) return false;
      file. write( ( char* ) &x_size, sizeof( int ) );
      file. write( ( char* ) &y_size, sizeof( int ) );
      file. write( ( char* ) &z_size, sizeof( int ) );
      if( file. bad() ) return false;
      return true;
   };

   //! Method for restoring the object from a file
   bool Load( istream& file )
   {
      if( ! tnlLongVector< T > :: Load( file ) ) return false;
      file. read( ( char* ) &x_size, sizeof( int ) );
      file. read( ( char* ) &y_size, sizeof( int ) );
      file. read( ( char* ) &z_size, sizeof( int ) );
      if( file. bad() ) return false;
      return true;
   };   
   
   protected:

   int x_size, y_size, z_size;
};

// Explicit instatiation
template class tnlField3D< double >;

#endif
