/***************************************************************************
                          tnlFieldSystem1D.h  -  description
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

#ifndef tnlFieldSystem1DH
#define tnlFieldSystem1DH

#include "tnlLongVector.h"
#include "tnlVector.h"

template< class T, int SYSTEM_SIZE, typename SYSTEM_INDEX > class tnlFieldSystem1D : public tnlLongVector< T >
{
   public:

   tnlFieldSystem1D()
   : tnlLongVector< T >( 0 )
   { };

   tnlFieldSystem1D( int _x_size )
   : tnlLongVector< T >( _x_size * SYSTEM_SIZE ),
     x_size( _x_size )
   { };

   tnlFieldSystem1D( const tnlFieldSystem1D& f )
   : tnlLongVector< T >( f ),
     x_size( f. x_size )
   { };

   tnlString GetType() const
   {
      T t;
      return tnlString( "tnlFieldSystem1D< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
   };

   int GetXSize() const
   {
      return x_size;
   };

   bool SetNewDimensions( int new_x_size )
   {
      x_size = new_x_size;
      return tnlLongVector< T > :: SetNewSize( x_size * SYSTEM_SIZE );
   };

   bool SetNewDimensions( const tnlFieldSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX >& f )
   {
      return SetNewDimensions( f. GetXSize() );
   };

   const T& operator() ( const SYSTEM_INDEX ind, const int i ) const
   {
      assert( i < x_size && i >= 0 && ( int ) ind < SYSTEM_SIZE );
      return tnlLongVector< T > :: data[ i * SYSTEM_SIZE + ind ];
   };

   T& operator() ( const SYSTEM_INDEX ind, const int i )
   {
      assert( i < x_size && i >= 0 && ( int ) ind < SYSTEM_SIZE );
      return tnlLongVector< T > :: data[ i * SYSTEM_SIZE + ind ];
   };

   tnlVector< SYSTEM_SIZE, T > operator() ( const int i ) const
   {
      assert( i < x_size && i >= 0 );
      tnlVector< SYSTEM_SIZE, T > v;
      int j;
      for( j = 0; j < SYSTEM_SIZE; j ++ )
         v[ j ] = ( *this )( ( SYSTEM_INDEX ) j, i );
      return v;
   };

   int GetLongVectorIndex( int i ) const
   {
      assert( i >= 0 && i < x_size );
      return i;
   };
   
   //! Method for saving the object to a file as a binary data
   bool Save( ostream& file ) const
   {
      if( ! tnlLongVector< T > :: Save( file ) ) return false;
      file. write( ( char* ) &x_size, sizeof( int ) );
      if( file. bad() ) return false;
      return true;
   };

   //! Method for restoring the object from a file
   bool Load( istream& file )
   {
      if( ! tnlLongVector< T > :: Load( file ) ) return false;
      file. read( ( char* ) &x_size, sizeof( int ) );
      if( file. bad() ) return false;
      return true;
   };   
   
   protected:

   int x_size;
};

// Explicit instatiation
//template class tnlFieldSystem1D< double, 1, int >;

#endif
