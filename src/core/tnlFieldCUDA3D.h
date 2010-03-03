/***************************************************************************
                          tnlFieldCUDA3D.h  -  description
                             -------------------
    begin                : 2010/01/12
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

#ifndef tnlFieldCUDA3DH
#define tnlFieldCUDA3DH

#include <string.h>
#include "tnlObject.h"
#include "tnlLongVectorCUDA.h"

template< class T > class tnlFieldCUDA3D : public tnlLongVectorCUDA< T >
{
   public:

   tnlFieldCUDA3D( const char* name = 0 )
   : tnlLongVectorCUDA< T >( name )
   { };

   tnlFieldCUDA3D( int _x_size,
             int _y_size,
             int _z_size )
   : tnlLongVectorCUDA< T >( _x_size * _y_size * _z_size ),
     x_size( _x_size ), y_size( _y_size ), z_size( _z_size )
   { };

   tnlFieldCUDA3D( const tnlFieldCUDA3D& f )
   : tnlLongVectorCUDA< T >( f ),
     x_size( f. x_size ), y_size( f. y_size ), z_size( f.z_size )
   { };

   tnlString GetType() const
   {
      T t;
      return tnlString( "tnlFieldCUDA3D< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
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
      return tnlLongVectorCUDA< T > :: SetNewSize( x_size * y_size * z_size );
   };

   bool SetNewDimensions( const tnlFieldCUDA3D< T >& f )
   {
      return SetNewDimensions( f. GetXSize(), f. GetYSize(), f. GetZSize() );
   };

   void SetSharedData( T* _data, const int _x_size, const int _y_size, const int _z_size )
   {
      tnlLongVectorCUDA< T > :: SetSharedData( _data, _x_size * _y_size * _z_size );
      x_size = _x_size;
      y_size = _y_size;
      z_size = _z_size;
   };
   
   protected:

   int x_size, y_size, z_size;
};

// Explicit instatiation
template class tnlFieldCUDA3D< double >;

#endif
