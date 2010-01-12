/***************************************************************************
                          tnlFieldCUDA1D.h  -  description
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

#ifndef tnlFieldCUDA1DH
#define tnlFieldCUDA1DH

#include <string.h>
#include "tnlObject.h"
#include "tnlLongVectorCUDA.h"

template< class T > class tnlFieldCUDA1D : public tnlLongVectorCUDA< T >
{
   public:

   tnlFieldCUDA1D()
   : tnlLongVectorCUDA< T >( 0 )
   { };

   tnlFieldCUDA1D( int _x_size )
   : tnlLongVectorCUDA< T >( _x_size ),
     x_size( _x_size )
   { };

   tnlFieldCUDA1D( const tnlFieldCUDA1D& f )
   : tnlLongVectorCUDA< T >( f ),
     x_size( f. x_size )
   { };

   tnlString GetType() const
   {
      T t;
      return tnlString( "tnlFieldCUDA1D< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
   };

   int GetXSize() const
   {
      return x_size;
   };

   bool SetNewDimensions( int new_x_size )
   {
      x_size = new_x_size;
      return tnlLongVectorCUDA< T > :: SetNewSize( x_size );
   };

   bool SetNewDimensions( const tnlFieldCUDA1D< T >& f )
   {
      return SetNewDimensions( f. GetXSize() );
   };
   
   protected:

   int x_size;
};

// Explicit instatiation
template class tnlFieldCUDA1D< double >;

#endif
