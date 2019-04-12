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

#include <TNL/param-types.h>
#include <TNL/ParallelFor.h>
#include <TNL/Containers/Algorithms/ArrayOperations.h>
#include <TNL/Containers/Algorithms/ArrayIO.h>
#include <TNL/Containers/Algorithms/ArrayAssignment.h>

#include "ArrayView.h"

namespace TNL {
namespace Containers {

template< typename Value,
          typename Device,
          typename Index >
String
ArrayView< Value, Device, Index >::
getType()
{
   return String( "Containers::ArrayView< " ) + ", " +
                  TNL::getType< Value >() + ", " +
                  Device::getDeviceType() + ", " +
                  TNL::getType< Index >() + " >";
}

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
typename ArrayView< Value, Device, Index >::ViewType
ArrayView< Value, Device, Index >::
getView()
{
   return *this;
}

template< typename Value,
          typename Device,
          typename Index >
typename ArrayView< Value, Device, Index >::ConstViewType
ArrayView< Value, Device, Index >::
getConstView() const
{
   return *this;
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
      Algorithms::ArrayOperations< Device >::copyMemory( getData(), view.getData(), getSize() );
   return *this;
}

template< typename Value,
          typename Device,
          typename Index >
   template< typename T >
ArrayView< Value, Device, Index >&
ArrayView< Value, Device, Index >::
operator = ( const T& data )
{
   Algorithms::ArrayAssignment< ArrayView, T >::assign( *this, data );
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
   return Algorithms::ArrayOperations< Device >::setMemoryElement( &data[ i ], value );
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
   return Algorithms::ArrayOperations< Device >::getMemoryElement( &data[ i ] );
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
   return Algorithms::ArrayOperations< DeviceType, typename ArrayT::DeviceType >::
            compareMemory( this->getData(),
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
setValue( Value value )
{
   TNL_ASSERT_GT( size, 0, "Attempted to set value to an empty array view." );
   Algorithms::ArrayOperations< Device >::setMemory( getData(), value, getSize() );
}

template< typename Value,
          typename Device,
          typename Index >
   template< typename Function >
void ArrayView< Value, Device, Index >::
evaluate( Function& f, const Index begin, Index end )
{
   TNL_ASSERT_TRUE( this->getData(), "Attempted to set a value of an empty array view." );

   ValueType* d = this->data;
   auto eval = [=] __cuda_callable__ ( Index i )
   {
      d[ i ] = f( i );
   };

   if( end == -1 )
      end = this->getSize();

   ParallelFor< DeviceType >::exec( begin, end, eval );
}

template< typename Value,
          typename Device,
          typename Index >
bool
ArrayView< Value, Device, Index >::
containsValue( Value value ) const
{
   return Algorithms::ArrayOperations< Device >::containsValue( data, size, value );
}

template< typename Value,
          typename Device,
          typename Index >
bool
ArrayView< Value, Device, Index >::
containsOnlyValue( Value value ) const
{
   return Algorithms::ArrayOperations< Device >::containsOnlyValue( data, size, value );
}

template< typename Value,
          typename Device,
          typename Index >
bool
ArrayView< Value, Device, Index >::
empty() const
{
   return ( data == nullptr );
}

template< typename Value,
          typename Device,
          typename Index >
void ArrayView< Value, Device, Index >::save( File& file ) const
{
   saveHeader( file, SerializationType::getType() );
   file.save( &this->size );
   Algorithms::ArrayIO< Value, Device, Index >::save( file, this->data, this->size );
}

template< typename Value,
          typename Device,
          typename Index >
void
ArrayView< Value, Device, Index >::
load( File& file )
{
   String type;
   loadHeader( file, type );
   if( type != SerializationType::getType() )
      throw Exceptions::FileDeserializationError( file.getFileName(), "invalid object type: " + type + " (expected " + SerializationType::getType() + ")." );
   Index _size;
   file.load( &_size );
   if( _size != this->getSize() )
      throw Exceptions::FileDeserializationError( file.getFileName(), "invalid array size: " + std::to_string(_size) + " (expected " + std::to_string( this->getSize() ) + ")." );
   Algorithms::ArrayIO< Value, Device, Index >::load( file, this->data, this->size );
}

template< typename Value,
          typename Device,
          typename Index >
void
ArrayView< Value, Device, Index >::
save( const String& fileName ) const
{
   File file;
   file.open( fileName, std::ios_base::out );
   this->save( file );
}

template< typename Value,
          typename Device,
          typename Index >
void
ArrayView< Value, Device, Index >::
load( const String& fileName )
{
   File file;
   file.open( fileName, std::ios_base::in );
   this->load( file );
}

template< typename Value, typename Device, typename Index >
std::ostream& operator<<( std::ostream& str, const ArrayView< Value, Device, Index >& v )
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
