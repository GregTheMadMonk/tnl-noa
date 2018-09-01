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
#include <TNL/File.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Containers/Algorithms/ArrayOperations.h>
#include <TNL/Math.h>
#include <TNL/param-types.h>

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
   TNL_ASSERT_FALSE( data | size, "ArrayView was initialized with a positive address and zero size or zero address and positive size." );
}

// initialization from other array containers (using shallow copy)
template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
ArrayView< Element, Device, Index >::
ArrayView( Array< Element, Device, Index >& array )
{
   this->bind( array );
}

template< typename Element,
          typename Device,
          typename Index >
   template< int Size >
__cuda_callable__
ArrayView< Element, Device, Index >::
ArrayView( StaticArray< Size, Element >& array )
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
   TNL_ASSERT_FALSE( data | size, "ArrayView was initialized with a positive address and zero size or zero address and positive size." );

   this->data = data;
   this->size = size;
}

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
void ArrayView< Element, Device, Index > :: bind( ArrayView& view )
{
   this->data = view.getData();
   this->size = view.getSize();
}

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
void
ArrayView< Element, Device, Index >::
bind( Array< Element, Device, Index >& array )
{
   this->data = array.getData();
   this->size = array.getSize();
}

template< typename Element,
          typename Device,
          typename Index >
   template< int Size >
__cuda_callable__
void
ArrayView< Element, Device, Index >::
bind( StaticArray< Size, Element >& array )
{
   this->data = array.getData();
   this->size = Size;
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
   TNL_ASSERT_EQ( size, view.size, "The sizes of the array views must be equal, views are not resizable." );
   if( size > 0 )
      Algorithms::ArrayOperations< Device > ::
         template copyMemory< Element, Element, Index >
                            ( getData(), view.getData(), getSize() );
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
String
ArrayView< Element, Device, Index >::
getTypeVirtual() const
{
   return getType();
}

template< typename Element,
          typename Device,
          typename Index >
String
ArrayView< Element, Device, Index >::
getSerializationType()
{
   return Array< Element, Device, Index >::getType();
}

template< typename Element,
          typename Device,
          typename Index >
String
ArrayView< Element, Device, Index >::
getSerializationTypeVirtual() const
{
   return getSerializationType();
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
   template< typename Device_ >
bool
ArrayView< Element, Device, Index >::
operator==( const ArrayView< Element, Device_, Index >& view ) const
{
   if( view.getSize() != getSize() )
      return false;
   return Algorithms::ArrayOperations< Device, Device_ >::
      template compareMemory< Element, Element, Index >
                            ( getData(), view.getData(), getSize() );
}

template< typename Element,
          typename Device,
          typename Index >
   template< typename Device_ >
bool
ArrayView< Element, Device, Index >::
operator!=( const ArrayView< Element, Device_, Index >& view ) const
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
   Algorithms::ArrayOperations< Device >::template setMemory< Element, Index >
                              ( getData(), value, getSize() );
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
ArrayView< Element, Device, Index >::
operator bool() const
{
   return data;
}

template< typename Element,
          typename Device,
          typename Index >
bool
ArrayView< Element, Device, Index >::
save( File& file ) const
{
   TNL_ASSERT( this->size != 0,
              std::cerr << "You try to save empty array." << std::endl );
   if( ! Object :: save( file ) )
      return false;
   if( ! file.write( &this->size ) )
      return false;
   if( ! file.write< Element, Device, Index >( this->data, this->size ) )
   {
      std::cerr << "I was not able to WRITE ArrayView with size " << this->getSize() << std::endl;
      return false;
   }
   return true;
}

template< typename Element,
          typename Device,
          typename Index >
bool
ArrayView< Element, Device, Index >::
save( const String& fileName ) const
{
   return Object::save( fileName );
}

template< typename Element,
          typename Device,
          typename Index >
bool
ArrayView< Element, Device, Index >::
load( File& file )
{
   if( ! Object :: load( file ) )
      return false;
   Index _size;
   if( ! file.read( &_size, 1 ) )
      return false;
   if( _size != this->size )
   {
      std::cerr << "Error: The size " << _size << " of the data to be load is different from the " <<
                   "allocated array. This is not possible in the shared array." << std::endl;
      return false;
   }
   if( _size )
   {
      if( ! file.read< Element, Device, Index >( this->data, this->size ) )
      {
         std::cerr << "I was not able to READ ArrayView with size " << this->getSize() << std::endl;
         return false;
      }
   }
   return true;
}

template< typename Element,
          typename Device,
          typename Index >
bool
ArrayView< Element, Device, Index >::
load( const String& fileName )
{
   return Object::load( fileName );
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
