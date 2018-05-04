/***************************************************************************
                          StaticEllpackIndexMultimapValues_impl.h  -  description
                             -------------------
    begin                : Sep 10, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include "StaticEllpackIndexMultimapValues.h"

#include <TNL/Assert.h>

namespace TNL {
 
template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int step >
__cuda_callable__
StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex, step >::
StaticEllpackIndexMultimapValues()
: values( nullptr )
{
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int step >
__cuda_callable__
StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex, step >::
StaticEllpackIndexMultimapValues( StaticEllpackIndexMultimapValues&& other )
: values( other.values )
{
   other.values = nullptr;
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int step >
__cuda_callable__
StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex, step >::
StaticEllpackIndexMultimapValues( IndexType* values )
: values( values )
{
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int step >
__cuda_callable__
StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex, step >&
StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex, step >::
operator=( const StaticEllpackIndexMultimapValues& other )
{
   if( this->values != other.values ) {
      for( LocalIndexType i = 0; i < this->getSize(); i++ )
         this->setValue( i, other[ i ] );
   }
   return *this;
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int step >
   template< typename Index_, typename LocalIndex_, int step_ >
__cuda_callable__
StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex, step >&
StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex, step >::
operator=( const StaticEllpackIndexMultimapValues< ValuesCount, Index_, Device, LocalIndex_, step_ >& other )
{
   for( LocalIndexType i = 0; i < this->getSize(); i++ )
      this->setValue( i, other[ i ] );
   return *this;
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int step >
__cuda_callable__
StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex, step >&
StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex, step >::
operator=( StaticEllpackIndexMultimapValues&& other )
{
   this->values = other.values;
   other.values = nullptr;
   return *this;
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int step >
__cuda_callable__
void
StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex, step >::
bind( const StaticEllpackIndexMultimapValues& other )
{
   this->values = other.values;
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int step >
constexpr LocalIndex
StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex, step >::
getSize() const
{
   return ValuesCount;
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int step >
constexpr LocalIndex
StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex, step >::
getAllocatedSize() const
{
   return ValuesCount;
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int step >
__cuda_callable__
void
StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex, step >::
setValue( const LocalIndexType& portIndex,
          const IndexType& value )
{
   TNL_ASSERT( this->values,
               std::cerr << "This instance is not bound to any multimap." << std::endl; );
   TNL_ASSERT( portIndex < this->getSize(),
               std::cerr << " portIndex = " << portIndex
                         << " getSize() = " << this->getSize()
                         << std::endl );
   this->values[ portIndex * step ] = value;
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int step >
__cuda_callable__
Index
StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex, step >::
getValue( const LocalIndexType& portIndex ) const
{
   TNL_ASSERT( this->values,
               std::cerr << "This instance is not bound to any multimap." << std::endl; );
   TNL_ASSERT( portIndex < this->getSize(),
               std::cerr << " portIndex = " << portIndex
                         << " getSize() = " << this->getSize()
                         << std::endl );
   return this->values[ portIndex * step ];
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int step >
__cuda_callable__
Index&
StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex, step >::
operator[]( const LocalIndexType& portIndex )
{
   TNL_ASSERT( this->values,
               std::cerr << "This instance is not bound to any multimap." << std::endl; );
   TNL_ASSERT( portIndex < this->getSize(),
               std::cerr << " portIndex = " << portIndex
                         << " getSize() = " << this->getSize()
                         << std::endl );
   return this->values[ portIndex * step ];
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int step >
__cuda_callable__
const Index&
StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex, step >::
operator[]( const LocalIndexType& portIndex ) const
{
   TNL_ASSERT( this->values,
               std::cerr << "This instance is not bound to any multimap." << std::endl; );
   TNL_ASSERT( portIndex < this->getSize(),
               std::cerr << " portIndex = " << portIndex
                         << " getSize() = " << this->getSize()
                         << std::endl );
   return this->values[ portIndex * step ];
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int step >
__cuda_callable__
bool
StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex, step >::
operator==( const StaticEllpackIndexMultimapValues& other ) const
{
   if( this->values == other.values )
      return true;
   if( ! this->values || ! other.values )
      return false;
   for( LocalIndexType i = 0; i < this->getSize(); i++ )
      if( this->operator[]( i ) != other[ i ] )
         return false;
   return true;
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int step >
__cuda_callable__
bool
StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex, step >::
operator!=( const StaticEllpackIndexMultimapValues& other ) const
{
   return ! ( *this == other );
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int step >
void
StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex, step >::
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

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int step >
std::ostream& operator << ( std::ostream& str, const StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex, step >& ports )
{
   ports.print( str );
   return str;
}

} // namespace TNL

