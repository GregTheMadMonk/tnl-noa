/***************************************************************************
                          tnlField1D.h  -  description
                             -------------------
    begin                : 2007/11/26
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

#ifndef tnlField1DH
#define tnlField1DH

#include <string.h>
#include "tnlObject.h"
#include "mLongVector.h"

template< class T > class tnlField1D : public mLongVector< T >
{
   public:

   tnlField1D()
   : mLongVector< T >( 0 )
   { };

   tnlField1D( long int _x_size )
   : mLongVector< T >( _x_size ),
     x_size( _x_size )
   { };

   tnlField1D( const tnlField1D& f )
   : mLongVector< T >( f ),
     x_size( f. x_size )
   { };

   tnlString GetType() const
   {
      T t;
      return tnlString( "tnlField1D< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
   };

   long int GetXSize() const
   {
      return x_size;
   };

   bool SetNewDimensions( long int new_x_size )
   {
      x_size = new_x_size;
      return mLongVector< T > :: SetNewSize( x_size );
   };

   bool SetNewDimensions( const tnlField1D< T >& f )
   {
      return SetNewDimensions( f. GetXSize() );
   };
   

   const T& operator() ( long int i ) const
   {
      assert( i < x_size && i >= 0 );
      return mLongVector< T > :: data[ i ];
   };

   T& operator() ( long int i )
   {
      assert( i < x_size && i >= 0 );
      return mLongVector< T > :: data[ i ];
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
template class tnlField1D< double >;

#endif
