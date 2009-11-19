/***************************************************************************
                          mFieldSystem1D.h  -  description
                             -------------------
    begin                : 2007/12/17
    copyright            : (C) 2007 by Tomá¹ Oberhuber
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

#ifndef mFieldSystem1DH
#define mFieldSystem1DH

#include "mLongVector.h"
#include "tnlVector.h"

template< class T, int SYSTEM_SIZE, typename SYSTEM_INDEX > class mFieldSystem1D : public mLongVector< T >
{
   public:

   mFieldSystem1D()
   : mLongVector< T >( 0 )
   { };

   mFieldSystem1D( long int _x_size )
   : mLongVector< T >( _x_size * SYSTEM_SIZE ),
     x_size( _x_size )
   { };

   mFieldSystem1D( const mFieldSystem1D& f )
   : mLongVector< T >( f ),
     x_size( f. x_size )
   { };

   tnlString GetType() const
   {
      T t;
      return tnlString( "mFieldSystem1D< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
   };

   long int GetXSize() const
   {
      return x_size;
   };

   bool SetNewDimensions( long int new_x_size )
   {
      x_size = new_x_size;
      return mLongVector< T > :: SetNewSize( x_size * SYSTEM_SIZE );
   };

   bool SetNewDimensions( const mFieldSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX >& f )
   {
      return SetNewDimensions( f. GetXSize() );
   };

   const T& operator() ( const SYSTEM_INDEX ind, const long int i ) const
   {
      assert( i < x_size && i >= 0 && ( long int ) ind < SYSTEM_SIZE );
      return mLongVector< T > :: data[ i * SYSTEM_SIZE + ind ];
   };

   T& operator() ( const SYSTEM_INDEX ind, const long int i )
   {
      assert( i < x_size && i >= 0 && ( long int ) ind < SYSTEM_SIZE );
      return mLongVector< T > :: data[ i * SYSTEM_SIZE + ind ];
   };

   tnlVector< SYSTEM_SIZE, T > operator() ( const long int i ) const
   {
      assert( i < x_size && i >= 0 );
      tnlVector< SYSTEM_SIZE, T > v;
      int j;
      for( j = 0; j < SYSTEM_SIZE; j ++ )
         v[ j ] = ( *this )( ( SYSTEM_INDEX ) j, i );
      return v;
   };

   long int GetLongVectorIndex( long int i ) const
   {
      assert( i >= 0 && i < x_size );
      return i;
   };
   
   //! Method for saving the object to a file as a binary data
   bool Save( ostream& file ) const
   {
      if( ! mLongVector< T > :: Save( file ) ) return false;
      file. write( ( char* ) &x_size, sizeof( long int ) );
      if( file. bad() ) return false;
      return true;
   };

   //! Method for restoring the object from a file
   bool Load( istream& file )
   {
      if( ! mLongVector< T > :: Load( file ) ) return false;
      file. read( ( char* ) &x_size, sizeof( long int ) );
      if( file. bad() ) return false;
      return true;
   };   
   
   protected:

   long int x_size;
};

// Explicit instatiation
//template class mFieldSystem1D< double, 1, int >;

#endif
