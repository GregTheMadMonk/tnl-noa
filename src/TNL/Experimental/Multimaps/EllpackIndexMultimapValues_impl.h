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
__cuda_callable__
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
EllpackIndexMultimapValues()
: values( nullptr ), valuesCount( nullptr ), allocatedSize( 0 )
{
}

template< typename Index,
          typename Device,
          typename LocalIndex >
__cuda_callable__
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
EllpackIndexMultimapValues( EllpackIndexMultimapValues&& other )
: values( other.values ), valuesCount( other.valuesCount ), allocatedSize( other.allocatedSize )
{
   other.values = nullptr;
   other.valuesCount = nullptr;
   other.allocatedSize = 0;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
__cuda_callable__
EllpackIndexMultimapValues< Index, Device, LocalIndex >&
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
operator=( const EllpackIndexMultimapValues& other )
{
   TNL_ASSERT( this->getSize() == other.getSize(), );
   if( this->values != other.values ) {
      for( LocalIndexType i = 0; i < this->getSize(); i++ )
         this->setValue( i, other[ i ] );
   }
   return *this;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
__cuda_callable__
EllpackIndexMultimapValues< Index, Device, LocalIndex >&
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
operator=( EllpackIndexMultimapValues&& other )
{
   this->values = other.values;
   this->valuesCount = other.valuesCount;
   this->allocatedSize = other.allocatedSize;
   other.values = nullptr;
   other.valuesCount = nullptr;
   other.allocatedSize = 0;
   return *this;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
__cuda_callable__
void
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
bind( const EllpackIndexMultimapValues& other )
{
   this->values = other.values;
   this->valuesCount = other.valuesCount;
   this->allocatedSize = other.allocatedSize;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
EllpackIndexMultimapValues( IndexType* values,
                            ValuesCountType* valuesCounts,
                            const IndexType& input,
                            const LocalIndexType& allocatedSize )
{
   this->values = &values[ input * allocatedSize ];
   this->valuesCount = &valuesCounts[ input ];
   this->allocatedSize = allocatedSize;
   TNL_ASSERT( *(this->valuesCount) <= allocatedSize, );
}

template< typename Index,
          typename Device,
          typename LocalIndex >
__cuda_callable__
bool
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
setSize( const LocalIndexType& size )
{
   if( ! this->valuesCount || size > this->allocatedSize )
      return false;
   *valuesCount = size;
   return true;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
__cuda_callable__
LocalIndex
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
getSize() const
{
   if( ! valuesCount )
      return 0;
   return *valuesCount;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
__cuda_callable__
LocalIndex
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
getAllocatedSize() const
{
   return this->allocatedSize;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
__cuda_callable__
void
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
setValue( const LocalIndexType& portIndex,
          const IndexType& value )
{
   TNL_ASSERT( portIndex < this->getSize(),
               std::cerr << " portIndex = " << portIndex
                         << " getSize() = " << this->getSize()
                         << std::endl );
   this->values[ portIndex ] = value;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
__cuda_callable__
Index
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
getValue( const LocalIndexType& portIndex ) const
{
   TNL_ASSERT( portIndex < this->getSize(),
               std::cerr << " portIndex = " << portIndex
                         << " getSize() = " << this->getSize()
                         << std::endl );
   return this->values[ portIndex ];
}

template< typename Index,
          typename Device,
          typename LocalIndex >
__cuda_callable__
Index&
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
operator[]( const LocalIndexType& portIndex )
{
   TNL_ASSERT( portIndex < this->getSize(),
               std::cerr << " portIndex = " << portIndex
                         << " getSize() = " << this->getSize()
                         << std::endl );
   return this->values[ portIndex ];
}

template< typename Index,
          typename Device,
          typename LocalIndex >
__cuda_callable__
const Index&
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
operator[]( const LocalIndexType& portIndex ) const
{
   TNL_ASSERT( portIndex < this->getSize(),
               std::cerr << " portIndex = " << portIndex
                         << " getSize() = " << this->getSize()
                         << std::endl );
   return this->values[ portIndex ];
}

template< typename Index,
          typename Device,
          typename LocalIndex >
__cuda_callable__
bool
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
operator==( const EllpackIndexMultimapValues& other ) const
{
   if( this->getSize() != other.getSize() )
      return false;
   for( LocalIndexType i = 0; i < this->getSize(); i++ )
      if( this->operator[]( i ) != other[ i ] )
         return false;
   return true;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
__cuda_callable__
bool
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
operator!=( const EllpackIndexMultimapValues& other ) const
{
   return ! ( *this == other );
}

template< typename Index,
          typename Device,
          typename LocalIndex >
void
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
print( std::ostream& str ) const
{
   str << "[ ";
   if( this->getSize() > 0 )
   {
      str << this->getValue( 0 );
      for( typename std::remove_const< Index >::type i = 1; i < this->getSize(); i++ )
         str << ", " << this->getValue( i );
   }
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

