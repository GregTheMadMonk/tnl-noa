/***************************************************************************
                          EllpackIndexMultimapValues_impl.h  -  description
                             -------------------
    begin                : Sep 10, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include "EllpackIndexMultimapValues.h"

#include <TNL/Assert.h>

namespace TNL {
 
template< typename Index,
          typename Device,
          typename LocalIndex >
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
EllpackIndexMultimapValues()
: values( nullptr ), allocatedSize( 0 )
{
}

template< typename Index,
          typename Device,
          typename LocalIndex >
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
EllpackIndexMultimapValues( ThisType&& other )
: values( other.values ), allocatedSize( other.allocatedSize )
{
   other.values = nullptr;
   other.allocatedSize = 0;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
EllpackIndexMultimapValues< Index, Device, LocalIndex >&
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
operator=( const ThisType& other )
{
   Assert( this->getSize() == other.getSize(), );
   if( this->values != other.values ) {
      for( LocalIndexType i = 0; i < this->getSize(); i++ )
         this->setValue( i, other[ i ] );
   }
   return *this;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
EllpackIndexMultimapValues< Index, Device, LocalIndex >&
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
operator=( ThisType&& other )
{
   this->values = other.values;
   this->allocatedSize = other.allocatedSize;
   other.values = nullptr;
   other.allocatedSize = 0;
   return *this;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
void
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
bind( const ThisType& other )
{
   this->values = other.values;
   this->allocatedSize = other.allocatedSize;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
EllpackIndexMultimapValues( IndexType* values,
                            const IndexType& input,
                            const LocalIndexType& allocatedSize )
{
   this->values = &values[ input * ( allocatedSize + 1 ) ];
   this->allocatedSize = allocatedSize;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
bool
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
setSize( const LocalIndexType& size )
{
   if( ! this->values || size > this->allocatedSize )
      return false;
   this->values[ this->allocatedSize ] = size;
   return true;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
LocalIndex
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
getSize() const
{
   if( ! this->values )
      return 0;
   return this->values[ this->allocatedSize ];
}

template< typename Index,
          typename Device,
          typename LocalIndex >
LocalIndex
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
getAllocatedSize() const
{
   return this->allocatedSize;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
void
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
setValue( const LocalIndexType& portIndex,
          const IndexType& value )
{
   Assert( portIndex < this->getSize(),
              std::cerr << " portIndex = " << portIndex
                        << " getSize() = " << this->getSize()
                        << std::endl );
   this->values[ portIndex ] = value;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
Index
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
getValue( const LocalIndexType& portIndex ) const
{
   Assert( portIndex < this->getSize(),
              std::cerr << " portIndex = " << portIndex
                        << " getSize() = " << this->getSize()
                        << std::endl );
   return this->values[ portIndex ];
}

template< typename Index,
          typename Device,
          typename LocalIndex >
Index&
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
operator[]( const LocalIndexType& portIndex )
{
   Assert( portIndex < this->getSize(),
              std::cerr << " portIndex = " << portIndex
                        << " getSize() = " << this->getSize()
                        << std::endl );
   return this->values[ portIndex ];
}

template< typename Index,
          typename Device,
          typename LocalIndex >
const Index&
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
operator[]( const LocalIndexType& portIndex ) const
{
   Assert( portIndex < this->getSize(),
              std::cerr << " portIndex = " << portIndex
                        << " getSize() = " << this->getSize()
                        << std::endl );
   return this->values[ portIndex ];
}

template< typename Index,
          typename Device,
          typename LocalIndex >
void
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
print( std::ostream& str ) const
{
   if( this->getSize() == 0 )
   {
      str << "[]";
      return;
   }
   str << "[ " << this->getValue( 0 );
   for( Index i = 1; i < this->getSize(); i++ )
      str << ", " << this->getValue( i );
   str << " ]";
}

template< typename Index,
          typename Device,
          typename LocalIndex >
std::ostream& operator << ( std::ostream& str, const EllpackIndexMultimapValues< Index, Device, LocalIndex >& ports )
{
   ports.print( str );
   return str;
}

} // namespace TNL

