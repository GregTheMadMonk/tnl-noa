/***************************************************************************
                          ArrayView_impl.h  -  description
                             -------------------
    begin                : Sep 1, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz et al.
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>

#include <TNL/param-types.h>
#include <TNL/Containers/Algorithms/ArrayOperations.h>

#include "ArrayView.h"

namespace TNL {
namespace Containers {

// explicit initialization by raw data pointer and size
template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
ArrayView< Element, Device, Index >::
ArrayView( Element* data, Index size ) : data(data), size(size)
{
   TNL_ASSERT_GE( size, 0, "ArrayView size was initialized with a negative size." );
   TNL_ASSERT_TRUE( (data == nullptr && size == 0) || (data != nullptr && size > 0),
                    "ArrayView was initialized with a positive address and zero size or zero address and positive size." );
}

// initialization from other array containers (using shallow copy)
template< typename Element,
          typename Device,
          typename Index >
   template< typename Element_ >
__cuda_callable__
ArrayView< Element, Device, Index >::
ArrayView( Array< Element_, Device, Index >& array )
{
   this->bind( array.getData(), array.getSize() );
}

template< typename Element,
          typename Device,
          typename Index >
   template< int Size, typename Element_ >
__cuda_callable__
ArrayView< Element, Device, Index >::
ArrayView( StaticArray< Size, Element_ >& array )
{
   this->bind( array );
}

// methods for rebinding (reinitialization)
template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
void
ArrayView< Element, Device, Index >::
bind( Element* data, Index size )
{
   TNL_ASSERT_GE( size, 0, "ArrayView size was initialized with a negative size." );
   TNL_ASSERT_TRUE( (data == nullptr && size == 0) || (data != nullptr && size > 0),
                    "ArrayView was initialized with a positive address and zero size or zero address and positive size." );

   this->data = data;
   this->size = size;
}

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
void ArrayView< Element, Device, Index >::bind( ArrayView& view )
{
   bind( view.getData(), view.getSize() );
}

template< typename Element,
          typename Device,
          typename Index >
   template< typename Element_ >
__cuda_callable__
void
ArrayView< Element, Device, Index >::
bind( Array< Element_, Device, Index >& array )
{
   bind( array.getData(), array.getSize() );
}

template< typename Element,
          typename Device,
          typename Index >
   template< int Size, typename Element_ >
__cuda_callable__
void
ArrayView< Element, Device, Index >::
bind( StaticArray< Size, Element_ >& array )
{
   bind( array.getData(), Size );
}


// Copy-assignment does deep copy, just like regular array, but the sizes
// must match (i.e. copy-assignment cannot resize).
template< typename Element,
           typename Device,
           typename Index >
ArrayView< Element, Device, Index >&
ArrayView< Element, Device, Index >::
operator=( const ArrayView& view )
{
   TNL_ASSERT_EQ( getSize(), view.getSize(), "The sizes of the array views must be equal, views are not resizable." );
   if( getSize() > 0 )
      Algorithms::ArrayOperations< Device >::copyMemory( getData(), view.getData(), getSize() );
   return *this;
}

template< typename Element,
           typename Device,
           typename Index >
   template< typename Element_, typename Device_, typename Index_ >
ArrayView< Element, Device, Index >&
ArrayView< Element, Device, Index >::
operator=( const ArrayView< Element_, Device_, Index_ >& view )
{
   TNL_ASSERT_EQ( getSize(), view.getSize(), "The sizes of the array views must be equal, views are not resizable." );
   if( getSize() > 0 )
      Algorithms::ArrayOperations< Device, Device_ >::copyMemory( getData(), view.getData(), getSize() );
   return *this;
}


template< typename Element,
          typename Device,
          typename Index >
String
ArrayView< Element, Device, Index >::
getType()
{
   return String( "Containers::ArrayView< " ) + ", " +
                  TNL::getType< Element >() + ", " +
                  Device::getDeviceType() + ", " +
                  TNL::getType< Index >() + " >";
}


template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
void
ArrayView< Element, Device, Index >::
swap( ArrayView& array )
{
   TNL::swap( data, array.data );
   TNL::swap( size, array.size );
}

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
void
ArrayView< Element, Device, Index >::
reset()
{
   data = nullptr;
   size = 0;
}


template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
const
Element* ArrayView< Element, Device, Index >::
getData() const
{
   return data;
}

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
Element*
ArrayView< Element, Device, Index >::
getData()
{
   return data;
}

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
Index
ArrayView< Element, Device, Index >::
getSize() const
{
   return size;
}

template< typename Element,
          typename Device,
          typename Index >
void
ArrayView< Element, Device, Index >::
setElement( Index i, Element value )
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, this->getSize(), "Element index is out of bounds." );
   return Algorithms::ArrayOperations< Device >::setMemoryElement( &data[ i ], value );
}

template< typename Element,
          typename Device,
          typename Index >
Element
ArrayView< Element, Device, Index >::
getElement( Index i ) const
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, this->getSize(), "Element index is out of bounds." );
   return Algorithms::ArrayOperations< Device >::getMemoryElement( &data[ i ] );
}

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
Element& ArrayView< Element, Device, Index >::
operator[]( Index i )
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, this->getSize(), "Element index is out of bounds." );
   return data[ i ];
}

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
const
Element& ArrayView< Element, Device, Index >::
operator[]( Index i ) const
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, this->getSize(), "Element index is out of bounds." );
   return data[ i ];
}

template< typename Element,
          typename Device,
          typename Index >
   template< typename Element_, typename Device_, typename Index_ >
bool
ArrayView< Element, Device, Index >::
operator==( const ArrayView< Element_, Device_, Index_ >& view ) const
{
   if( view.getSize() != getSize() )
      return false;
   if( getSize() == 0 )
      return true;
   return Algorithms::ArrayOperations< Device, Device_ >::compareMemory( getData(), view.getData(), getSize() );
}

template< typename Element,
          typename Device,
          typename Index >
   template< typename Element_, typename Device_, typename Index_ >
bool
ArrayView< Element, Device, Index >::
operator!=( const ArrayView< Element_, Device_, Index_ >& view ) const
{
   return ! ( *this == view );
}

template< typename Element,
          typename Device,
          typename Index >
void
ArrayView< Element, Device, Index >::
setValue( Element value )
{
   TNL_ASSERT_GT( size, 0, "Attempted to set value to an empty array view." );
   Algorithms::ArrayOperations< Device >::setMemory( getData(), value, getSize() );
}

template< typename Element,
          typename Device,
          typename Index >
bool
ArrayView< Element, Device, Index >::
containsValue( Element value ) const
{
   return Algorithms::ArrayOperations< Device >::containsValue( data, size, value );
}

template< typename Element,
          typename Device,
          typename Index >
bool
ArrayView< Element, Device, Index >::
containsOnlyValue( Element value ) const
{
   return Algorithms::ArrayOperations< Device >::containsOnlyValue( data, size, value );
}

template< typename Element,
          typename Device,
          typename Index >
ArrayView< Element, Device, Index >::
operator bool() const
{
   return data;
}


template< typename Element, typename Device, typename Index >
std::ostream& operator<<( std::ostream& str, const ArrayView< Element, Device, Index >& v )
{
   str << "[ ";
   if( v.getSize() > 0 )
   {
      str << v.getElement( 0 );
      for( Index i = 1; i < v.getSize(); i++ )
         str << ", " << v.getElement( i );
   }
   str << " ]";
   return str;
}

} // namespace Containers
} // namespace TNL
