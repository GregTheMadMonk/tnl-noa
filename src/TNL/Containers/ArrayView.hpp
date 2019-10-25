/***************************************************************************
                          ArrayView_impl.h  -  description
                             -------------------
    begin                : Sep 1, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>
#include <stdexcept>

#include <TNL/TypeInfo.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Algorithms/MemoryOperations.h>
#include <TNL/Algorithms/MultiDeviceMemoryOperations.h>
#include <TNL/Containers/detail/ArrayIO.h>
#include <TNL/Containers/detail/ArrayAssignment.h>

#include "ArrayView.h"

namespace TNL {
namespace Containers {

// explicit initialization by raw data pointer and size
template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
ArrayView< Value, Device, Index >::
ArrayView( Value* data, Index size ) : data(data), size(size)
{
   TNL_ASSERT_GE( size, 0, "ArrayView size was initialized with a negative size." );
   TNL_ASSERT_TRUE( (data == nullptr && size == 0) || (data != nullptr && size > 0),
                    "ArrayView was initialized with a positive address and zero size or zero address and positive size." );
}

// methods for rebinding (reinitialization)
template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
void
ArrayView< Value, Device, Index >::
bind( Value* data, Index size )
{
   TNL_ASSERT_GE( size, 0, "ArrayView size was initialized with a negative size." );
   TNL_ASSERT_TRUE( (data == nullptr && size == 0) || (data != nullptr && size > 0),
                    "ArrayView was initialized with a positive address and zero size or zero address and positive size." );

   this->data = data;
   this->size = size;
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
void ArrayView< Value, Device, Index >::bind( ArrayView view )
{
   bind( view.getData(), view.getSize() );
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
typename ArrayView< Value, Device, Index >::ViewType
ArrayView< Value, Device, Index >::
getView( const IndexType begin, IndexType end )
{
   if( end == 0 )
      end = this->getSize();
   return ViewType( getData() + begin, end - begin );;
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
typename ArrayView< Value, Device, Index >::ConstViewType
ArrayView< Value, Device, Index >::
getConstView( const IndexType begin, IndexType end ) const
{
   if( end == 0 )
      end = this->getSize();
   return ConstViewType( getData() + begin, end - begin );
}

// Copy-assignment does deep copy, just like regular array, but the sizes
// must match (i.e. copy-assignment cannot resize).
template< typename Value,
           typename Device,
           typename Index >
ArrayView< Value, Device, Index >&
ArrayView< Value, Device, Index >::
operator=( const ArrayView& view )
{
   TNL_ASSERT_EQ( getSize(), view.getSize(), "The sizes of the array views must be equal, views are not resizable." );
   if( getSize() > 0 )
      Algorithms::MemoryOperations< Device >::copy( getData(), view.getData(), getSize() );
   return *this;
}

template< typename Value,
          typename Device,
          typename Index >
   template< typename T, typename..., typename >
ArrayView< Value, Device, Index >&
ArrayView< Value, Device, Index >::
operator=( const T& data )
{
   detail::ArrayAssignment< ArrayView, T >::assign( *this, data );
   return *this;
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
void
ArrayView< Value, Device, Index >::
swap( ArrayView& array )
{
   TNL::swap( data, array.data );
   TNL::swap( size, array.size );
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
void
ArrayView< Value, Device, Index >::
reset()
{
   data = nullptr;
   size = 0;
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
bool
ArrayView< Value, Device, Index >::
empty() const
{
   return data == nullptr;
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
const
Value* ArrayView< Value, Device, Index >::
getData() const
{
   return data;
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
Value*
ArrayView< Value, Device, Index >::
getData()
{
   return data;
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
const
Value* ArrayView< Value, Device, Index >::
getArrayData() const
{
   return data;
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
Value*
ArrayView< Value, Device, Index >::
getArrayData()
{
   return data;
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
Index
ArrayView< Value, Device, Index >::
getSize() const
{
   return size;
}

template< typename Value,
          typename Device,
          typename Index >
void
ArrayView< Value, Device, Index >::
setElement( Index i, Value value )
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, this->getSize(), "Element index is out of bounds." );
   return Algorithms::MemoryOperations< Device >::setElement( &data[ i ], value );
}

template< typename Value,
          typename Device,
          typename Index >
Value
ArrayView< Value, Device, Index >::
getElement( Index i ) const
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, this->getSize(), "Element index is out of bounds." );
   return Algorithms::MemoryOperations< Device >::getElement( &data[ i ] );
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
Value& ArrayView< Value, Device, Index >::
operator[]( Index i )
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, this->getSize(), "Element index is out of bounds." );
   return data[ i ];
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
const
Value& ArrayView< Value, Device, Index >::
operator[]( Index i ) const
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, this->getSize(), "Element index is out of bounds." );
   return data[ i ];
}

template< typename Value,
          typename Device,
          typename Index >
   template< typename ArrayT >
bool
ArrayView< Value, Device, Index >::
operator==( const ArrayT& array ) const
{
   if( array.getSize() != this->getSize() )
      return false;
   if( this->getSize() == 0 )
      return true;
   return Algorithms::MultiDeviceMemoryOperations< DeviceType, typename ArrayT::DeviceType >::
            compare( this->getData(),
                           array.getData(),
                           array.getSize() );
}

template< typename Value,
          typename Device,
          typename Index >
   template< typename ArrayT >
bool
ArrayView< Value, Device, Index >::
operator!=( const ArrayT& array ) const
{
   return ! ( *this == array );
}

template< typename Value,
          typename Device,
          typename Index >
void
ArrayView< Value, Device, Index >::
setValue( Value value, const Index begin, Index end )
{
   TNL_ASSERT_GT( size, 0, "Attempted to set value to an empty array view." );
   if( end == 0 )
      end = this->getSize();
   Algorithms::MemoryOperations< Device >::set( &getData()[ begin ], value, end - begin );
}

template< typename Value,
          typename Device,
          typename Index >
   template< typename Function >
void ArrayView< Value, Device, Index >::
evaluate( const Function& f, const Index begin, Index end )
{
   TNL_ASSERT_TRUE( this->getData(), "Attempted to set a value of an empty array view." );

   ValueType* d = this->data;
   auto eval = [=] __cuda_callable__ ( Index i )
   {
      d[ i ] = f( i );
   };

   if( end == 0 )
      end = this->getSize();

   Algorithms::ParallelFor< DeviceType >::exec( begin, end, eval );
}

template< typename Value,
          typename Device,
          typename Index >
bool
ArrayView< Value, Device, Index >::
containsValue( Value value,
               const Index begin,
               Index end ) const
{
   if( end == 0 )
      end = this->getSize();
   return Algorithms::MemoryOperations< Device >::containsValue( &this->getData()[ begin ], end - begin, value );
}

template< typename Value,
          typename Device,
          typename Index >
bool
ArrayView< Value, Device, Index >::
containsOnlyValue( Value value,
                   const Index begin,
                   Index end  ) const
{
   if( end == 0 )
      end = this->getSize();
   return Algorithms::MemoryOperations< Device >::containsOnlyValue( &this->getData()[ begin ], end - begin, value );
}

template< typename Value, typename Device, typename Index >
std::ostream& operator<<( std::ostream& str, const ArrayView< Value, Device, Index >& view )
{
   str << "[ ";
   if( view.getSize() > 0 )
   {
      str << view.getElement( 0 );
      for( Index i = 1; i < view.getSize(); i++ )
         str << ", " << view.getElement( i );
   }
   str << " ]";
   return str;
}

template< typename Value,
          typename Device,
          typename Index >
void ArrayView< Value, Device, Index >::save( const String& fileName ) const
{
   File( fileName, std::ios_base::out ) << *this;
}

template< typename Value,
          typename Device,
          typename Index >
void
ArrayView< Value, Device, Index >::
load( const String& fileName )
{
   File( fileName, std::ios_base::in ) >> *this;
}

// Serialization of array views into binary files.
template< typename Value, typename Device, typename Index >
File& operator<<( File& file, const ArrayView< Value, Device, Index > view )
{
   using IO = detail::ArrayIO< Value, Device, Index >;
   saveObjectType( file, IO::getSerializationType() );
   const Index size = view.getSize();
   file.save( &size );
   IO::save( file, view.getData(), view.getSize() );
   return file;
}

template< typename Value, typename Device, typename Index >
File& operator<<( File&& file, const ArrayView< Value, Device, Index > view )
{
   File& f = file;
   return f << view;
}

// Deserialization of array views from binary files.
template< typename Value, typename Device, typename Index >
File& operator>>( File& file, ArrayView< Value, Device, Index > view )
{
   using IO = detail::ArrayIO< Value, Device, Index >;
   const String type = getObjectType( file );
   if( type != IO::getSerializationType() )
      throw Exceptions::FileDeserializationError( file.getFileName(), "object type does not match (expected " + IO::getSerializationType() + ", found " + type + ")." );
   Index _size;
   file.load( &_size );
   if( _size != view.getSize() )
      throw Exceptions::FileDeserializationError( file.getFileName(), "invalid array size: " + std::to_string(_size) + " (expected " + std::to_string( view.getSize() ) + ")." );
   IO::load( file, view.getData(), view.getSize() );
   return file;
}

template< typename Value, typename Device, typename Index >
File& operator>>( File&& file, ArrayView< Value, Device, Index > view )
{
   File& f = file;
   return f >> view;
}

} // namespace Containers
} // namespace TNL
