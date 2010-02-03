/***************************************************************************
                          tnlField2D.h  -  description
                             -------------------
    begin                : 2010/01/12
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

#ifndef tnlFieldCUDA2DH
#define tnlFieldCUDA2DH

#include <string.h>
#include <core/tnlObject.h>
#include <core/tnlLongVectorCUDA.h>
#include <core/tnlField2D.h>

template< class T > class tnlFieldCUDA2D : public tnlLongVectorCUDA< T >
{
   public:

   tnlFieldCUDA2D()
   : tnlLongVectorCUDA< T >( 0 )
   { };

   tnlFieldCUDA2D( int _x_size,
               int _y_size )
   : tnlLongVectorCUDA< T >( _x_size * _y_size ),
     x_size( _x_size ), y_size( _y_size )
   { };

   tnlFieldCUDA2D( const tnlFieldCUDA2D& f )
   : tnlLongVectorCUDA< T >( f ),
     x_size( f. x_size ), y_size( f. y_size )
   { };

   tnlFieldCUDA2D( const tnlField2D< T >& f )
      : tnlLongVectorCUDA< T >( f ),
        x_size( f. GetXSize() ), y_size( f. GetYSize() )
      { };

   tnlString GetType() const
   {
      T t;
      return tnlString( "tnlFieldCUDA2D< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
   };

   int GetXSize() const
   {
      return x_size;
   };

   int GetYSize() const
   {
      return y_size;
   };

   bool SetNewDimensions( int new_x_size,
                          int new_y_size )
   {
      x_size = new_x_size;
      y_size = new_y_size;
      return tnlLongVectorCUDA< T > :: SetNewSize( x_size * y_size );
   };

   bool SetNewDimensions( const tnlFieldCUDA2D< T >& f )
   {
      return SetNewDimensions( f. GetXSize(), f. GetYSize() );
   };

   void SetSharedData( T* _data, const int _x_size, const int _y_size )
   {
      tnlLongVectorCUDA< T > :: SetSharedData( _data, _x_size * _y_size );
      x_size = _x_size;
      y_size = _y_size;
   };
   
   bool copyFrom( const tnlField2D< T >& host_field )
   {
	   return tnlLongVectorCUDA< T > :: copyFrom( ( const tnlLongVector< T >& ) host_field );
   }

   protected:

   int x_size, y_size;
};

template< class T > bool tnlField2D< T > :: copyFrom( const tnlFieldCUDA2D< T >& device_field )
{
	return tnlLongVector< T > :: copyFrom( ( const tnlLongVectorCUDA< T >&) device_field );
}
// Explicit instatiation
template class tnlFieldCUDA2D< double >;

#endif
