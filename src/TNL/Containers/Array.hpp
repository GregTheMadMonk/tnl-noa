/***************************************************************************
                          Array_impl.h  -  description
                             -------------------
    begin                : Nov 8, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>
#include <stdexcept>

#include <TNL/Assert.h>
#include <TNL/Math.h>
#include <TNL/param-types.h>
#include <TNL/Containers/Algorithms/ArrayOperations.h>
#include <TNL/Containers/Algorithms/ArrayIO.h>
#include <TNL/Containers/Algorithms/ArrayAssignment.h>

#include "Array.h"

namespace TNL {
namespace Containers {

template< typename Value,
          typename Device,
          typename Index >
Array< Value, Device, Index >::
Array()
: size( 0 ),
  data( 0 ),
  allocationPointer( 0 ),
  referenceCounter( 0 )
{
}

template< typename Value,
          typename Device,
          typename Index >
Array< Value, Device, Index >::
Array( const IndexType& size )
: size( 0 ),
  data( 0 ),
  allocationPointer( 0 ),
  referenceCounter( 0 )
{
   this->setSize( size );
}

template< typename Value,
          typename Device,
          typename Index >
Array< Value, Device, Index >::
Array( Value* data,
       const IndexType& size )
: size( 0 ),
  data( nullptr ),
  allocationPointer( nullptr ),
  referenceCounter( 0 )
{
   this->setSize( size );
   Algorithms::ArrayOperations< Device >::copyMemory( this->getData(), data, size );
}

template< typename Value,
          typename Device,
          typename Index >
Array< Value, Device, Index >::
Array( const Array< Value, Device, Index >& array )
: size( 0 ),
  data( nullptr ),
  allocationPointer( nullptr ),
  referenceCounter( 0 )
{
   this->setSize( array.getSize() );
   Algorithms::ArrayOperations< Device >::copyMemory( this->getData(), array.getData(), array.getSize() );
}

template< typename Value,
          typename Device,
          typename Index >
Array< Value, Device, Index >::
Array( const Array< Value, Device, Index >& array,
       IndexType begin,
       IndexType size )
: size( size ),
  data( nullptr ),
  allocationPointer( nullptr ),
  referenceCounter( 0 )
{
   if( size == 0 )
      size = array.getSize() - begin;
   TNL_ASSERT_LT( begin, array.getSize(), "Begin of array is out of bounds." );
   TNL_ASSERT_LE( begin + size, array.getSize(), "End of array is out of bounds." );

   this->setSize( size );
   Algorithms::ArrayOperations< Device >::copyMemory( this->getData(), &array.getData()[ begin ], size );
}

template< typename Value,
          typename Device,
          typename Index >
   template< typename InValue >
Array< Value, Device, Index >::
Array( const std::initializer_list< InValue >& list )
: size( 0 ),
  data( 0 ),
  allocationPointer( 0 ),
  referenceCounter( 0 )
{
   this->setSize( list.size() );
   // Here we assume that the underlying array for std::initializer_list is
   // const T[N] as noted here:
   // https://en.cppreference.com/w/cpp/utility/initializer_list
   Algorithms::ArrayOperations< Device, Devices::Host >::copyMemory( this->getData(), &( *list.begin() ), list.size() );
}

template< typename Value,
          typename Device,
          typename Index >
   template< typename InValue >
Array< Value, Device, Index >::
Array( const std::list< InValue >& list )
: size( 0 ),
  data( 0 ),
  allocationPointer( 0 ),
  referenceCounter( 0 )
{
   this->setSize( list.size() );
   Algorithms::ArrayOperations< Device >::copySTLList( this->getData(), list );
}

template< typename Value,
          typename Device,
          typename Index >
   template< typename InValue >
Array< Value, Device, Index >::
Array( const std::vector< InValue >& vector )
: size( 0 ),
  data( 0 ),
  allocationPointer( 0 ),
  referenceCounter( 0 )
{
   this->setSize( vector.size() );
   Algorithms::ArrayOperations< Device, Devices::Host >::copyMemory( this->getData(), vector.data(), vector.size() );
}

template< typename Value,
          typename Device,
          typename Index >
String
Array< Value, Device, Index >::
getType()
{
   return String( "Containers::Array< " ) +
          TNL::getType< Value >() + ", " +
          Device::getDeviceType() + ", " +
          TNL::getType< Index >() + " >";
}

template< typename Value,
          typename Device,
          typename Index >
String
Array< Value, Device, Index >::
getTypeVirtual() const
{
   return this->getType();
}

template< typename Value,
          typename Device,
          typename Index >
String
Array< Value, Device, Index >::
getSerializationType()
{
   return HostType::getType();
}

template< typename Value,
          typename Device,
          typename Index >
String
Array< Value, Device, Index >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Value,
          typename Device,
          typename Index >
void
Array< Value, Device, Index >::
releaseData() const
{
   if( this->referenceCounter )
   {
      if( --*this->referenceCounter == 0 )
      {
         Algorithms::ArrayOperations< Device >::freeMemory( this->allocationPointer );
         delete this->referenceCounter;
         //std::cerr << "Deallocating reference counter " << this->referenceCounter << std::endl;
      }
   }
   else
      if( allocationPointer )
         Algorithms::ArrayOperations< Device >::freeMemory( this->allocationPointer );
   this->allocationPointer = 0;
   this->data = 0;
   this->size = 0;
   this->referenceCounter = 0;
}

template< typename Value,
          typename Device,
          typename Index >
void
Array< Value, Device, Index >::
setSize( Index size )
{
   TNL_ASSERT_GE( size, (Index) 0, "Array size must be non-negative." );

   if( this->size == size && allocationPointer && ! referenceCounter )
      return;
   this->releaseData();

   // Allocating zero bytes is useless. Moreover, the allocators don't behave the same way:
   // "operator new" returns some non-zero address, the latter returns a null pointer.
   if( size > 0 ) {
      Algorithms::ArrayOperations< Device >::allocateMemory( this->allocationPointer, size );
      this->data = this->allocationPointer;
      this->size = size;
      TNL_ASSERT_TRUE( this->allocationPointer,
                       "This should never happen - allocator did not throw on an error." );
   }
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
Index
Array< Value, Device, Index >::
getSize() const
{
   return this->size;
}

template< typename Value,
          typename Device,
          typename Index >
   template< typename ArrayT >
void
Array< Value, Device, Index >::
setLike( const ArrayT& array )
{
   setSize( array.getSize() );
}

template< typename Value,
          typename Device,
          typename Index >
void
Array< Value, Device, Index >::
bind( Value* data,
      const Index size )
{
   TNL_ASSERT_TRUE( data, "Null pointer cannot be bound." );
   this->releaseData();
   this->data = data;
   this->size = size;
}

template< typename Value,
          typename Device,
          typename Index >
   template< typename ArrayT >
void
Array< Value, Device, Index >::
bind( const ArrayT& array,
      const IndexType& begin,
      const IndexType& size )
{
   // all template parameters of Array must match, otherwise binding does not make sense
   static_assert( std::is_same< Value, typename ArrayT::ValueType >::value, "ValueType of both arrays must be the same." );
   static_assert( std::is_same< Device, typename ArrayT::DeviceType >::value, "DeviceType of both arrays must be the same." );
   static_assert( std::is_same< Index, typename ArrayT::IndexType >::value, "IndexType of both arrays must be the same." );
   TNL_ASSERT_TRUE( array.getData(), "Empty array cannot be bound." );
   TNL_ASSERT_LT( begin, array.getSize(), "Begin of array is out of bounds." );
   TNL_ASSERT_LE( begin + size, array.getSize(), "End of array is out of bounds." );

   this->releaseData();
   if( size )
      this->size = size;
   else
      this->size = array.getSize() - begin;
   this->data = const_cast< Value* >( &array.getData()[ begin ] );
   this->allocationPointer = array.allocationPointer;
   if( array.allocationPointer )
   {
      if( array.referenceCounter )
      {
         this->referenceCounter = array.referenceCounter;
         ( *this->referenceCounter )++;
      }
      else
      {
         this->referenceCounter = array.referenceCounter = new int( 2 );
         //std::cerr << "Allocating reference counter " << this->referenceCounter << std::endl;
      }
   }
}

template< typename Value,
          typename Device,
          typename Index >
   template< int Size >
void
Array< Value, Device, Index >::
bind( StaticArray< Size, Value >& array )
{
   this->releaseData();
   this->size = Size;
   this->data = array.getData();
}

template< typename Value,
          typename Device,
          typename Index >
typename Array< Value, Device, Index >::ViewType
Array< Value, Device, Index >::
getView( IndexType begin, IndexType end )
{
   if( end == 0 )
      end = getSize();
   return ViewType( getData() + begin, end - begin );
}

template< typename Value,
          typename Device,
          typename Index >
typename Array< Value, Device, Index >::ConstViewType
Array< Value, Device, Index >::
getConstView( IndexType begin, IndexType end ) const
{
   if( end == 0 )
      end = getSize();
   return ConstViewType( getData() + begin, end - begin );
}

template< typename Value,
          typename Device,
          typename Index >
Array< Value, Device, Index >::
operator ViewType()
{
   return getView();
}

template< typename Value,
          typename Device,
          typename Index >
Array< Value, Device, Index >::
operator ConstViewType() const
{
   return getConstView();
}

template< typename Value,
          typename Device,
          typename Index >
void
Array< Value, Device, Index >::
swap( Array< Value, Device, Index >& array )
{
   TNL::swap( this->size, array.size );
   TNL::swap( this->data, array.data );
   TNL::swap( this->allocationPointer, array.allocationPointer );
   TNL::swap( this->referenceCounter, array.referenceCounter );
}

template< typename Value,
          typename Device,
          typename Index >
void
Array< Value, Device, Index >::
reset()
{
   this->releaseData();
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
const Value*
Array< Value, Device, Index >::
getData() const
{
   return this->data;
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
Value*
Array< Value, Device, Index >::
getData()
{
   return this->data;
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
const Value*
Array< Value, Device, Index >::
getArrayData() const
{
   return this->data;
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
Value*
Array< Value, Device, Index >::
getArrayData()
{
   return this->data;
}

template< typename Value,
          typename Device,
          typename Index >
void
Array< Value, Device, Index >::
setElement( const Index& i, const Value& x )
{
   TNL_ASSERT_GE( i, (Index) 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, this->getSize(), "Element index is out of bounds." );
   return Algorithms::ArrayOperations< Device >::setMemoryElement( &( this->data[ i ] ), x );
}

template< typename Value,
          typename Device,
          typename Index >
Value
Array< Value, Device, Index >::
getElement( const Index& i ) const
{
   TNL_ASSERT_GE( i, (Index) 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, this->getSize(), "Element index is out of bounds." );
   return Algorithms::ArrayOperations< Device >::getMemoryElement( & ( this->data[ i ] ) );
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
inline Value&
Array< Value, Device, Index >::
operator[]( const Index& i )
{
   TNL_ASSERT_GE( i, (Index) 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, this->getSize(), "Element index is out of bounds." );
   return this->data[ i ];
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
inline const Value&
Array< Value, Device, Index >::
operator[]( const Index& i ) const
{
   TNL_ASSERT_GE( i, (Index) 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, this->getSize(), "Element index is out of bounds." );
   return this->data[ i ];
}

template< typename Value,
          typename Device,
          typename Index >
Array< Value, Device, Index >&
Array< Value, Device, Index >::
operator=( const Array< Value, Device, Index >& array )
{
   //TNL_ASSERT_EQ( array.getSize(), this->getSize(), "Array sizes must be the same." );
   if( this->getSize() != array.getSize() )
      this->setLike( array );
   if( this->getSize() > 0 )
      Algorithms::ArrayOperations< Device >::
         copyMemory( this->getData(),
                     array.getData(),
                     array.getSize() );
   return *this;
}

template< typename Value,
          typename Device,
          typename Index >
Array< Value, Device, Index >&
Array< Value, Device, Index >::
operator=( Array< Value, Device, Index >&& array )
{
   reset();

   this->size = array.size;
   this->data = array.data;
   this->allocationPointer = array.allocationPointer;
   this->referenceCounter = array.referenceCounter;

   array.size = 0;
   array.data = nullptr;
   array.allocationPointer = nullptr;
   array.referenceCounter = nullptr;
   return *this;
}


template< typename Value,
          typename Device,
          typename Index >
   template< typename T >
Array< Value, Device, Index >&
Array< Value, Device, Index >::
operator=( const T& data )
{
   Algorithms::ArrayAssignment< Array, T >::resize( *this, data );
   Algorithms::ArrayAssignment< Array, T >::assign( *this, data );
   return *this;
}

template< typename Value,
          typename Device,
          typename Index >
   template< typename InValue >
Array< Value, Device, Index >&
Array< Value, Device, Index >::
operator=( const std::list< InValue >& list )
{
   this->setSize( list.size() );
   Algorithms::ArrayOperations< Device >::copySTLList( this->getData(), list );
   return *this;
}

template< typename Value,
          typename Device,
          typename Index >
   template< typename InValue >
Array< Value, Device, Index >&
Array< Value, Device, Index >::
operator=( const std::vector< InValue >& vector )
{
   if( this->getSize() != vector.size() )
      this->setSize( vector.size() );
   Algorithms::ArrayOperations< Device, Devices::Host >::copyMemory( this->getData(), vector.data(), vector.size() );
   return *this;
}

template< typename Value,
          typename Device,
          typename Index >
   template< typename ArrayT >
bool
Array< Value, Device, Index >::
operator==( const ArrayT& array ) const
{
   if( array.getSize() != this->getSize() )
      return false;
   if( this->getSize() == 0 )
      return true;
   return Algorithms::ArrayOperations< Device, typename ArrayT::DeviceType >::
            compareMemory( this->getData(),
                           array.getData(),
                           array.getSize() );
}

template< typename Value,
          typename Device,
          typename Index >
   template< typename ArrayT >
bool
Array< Value, Device, Index >::
operator!=( const ArrayT& array ) const
{
   return ! ( *this == array );
}

template< typename Value,
          typename Device,
          typename Index >
void
Array< Value, Device, Index >::
setValue( const ValueType& v,
          IndexType begin,
          IndexType end )
{
   TNL_ASSERT_TRUE( this->getData(), "Attempted to set a value of an empty array." );
   if( end == 0 )
      end = this->getSize();
   Algorithms::ArrayOperations< Device >::setMemory( &this->getData()[ begin ], v, end - begin );
}

template< typename Value,
          typename Device,
          typename Index >
   template< typename Function >
void
Array< Value, Device, Index >::
evaluate( const Function& f,
          IndexType begin,
          IndexType end )
{
   TNL_ASSERT_TRUE( this->getData(), "Attempted to set a value of an empty array." );
   this->getView().evaluate( f, begin, end );
}

template< typename Value,
          typename Device,
          typename Index >
bool
Array< Value, Device, Index >::
containsValue( const ValueType& v,
               IndexType begin,
               IndexType end ) const
{
   TNL_ASSERT_TRUE( this->getData(), "Attempted to check a value of an empty array." );
   if( end == 0 )
      end = this->getSize();

   return Algorithms::ArrayOperations< Device >::containsValue( &this->getData()[ begin ], end - begin, v );
}

template< typename Value,
          typename Device,
          typename Index >
bool
Array< Value, Device, Index >::
containsOnlyValue( const ValueType& v,
                   IndexType begin,
                   IndexType end ) const
{
   TNL_ASSERT_TRUE( this->getData(), "Attempted to check a value of an empty array." );
   if( end == 0 )
      end = this->getSize();

   return Algorithms::ArrayOperations< Device >::containsOnlyValue( &this->getData()[ begin ], end - begin, v );
}

template< typename Value,
          typename Device,
          typename Index >
bool
__cuda_callable__
Array< Value, Device, Index >::
empty() const
{
   return ( data == nullptr );
}

template< typename Value,
          typename Device,
          typename Index >
void
Array< Value, Device, Index >::
save( const String& fileName ) const
{
   File( fileName, std::ios_base::out ) << *this;
}

template< typename Value,
          typename Device,
          typename Index >
void
Array< Value, Device, Index >::
load( const String& fileName )
{
   File( fileName, std::ios_base::in ) >> *this;
}

template< typename Value,
          typename Device,
          typename Index >
Array< Value, Device, Index >::
~Array()
{
   this->releaseData();
}

template< typename Value, typename Device, typename Index >
std::ostream& operator<<( std::ostream& str, const Array< Value, Device, Index >& array )
{
   str << "[ ";
   if( array.getSize() > 0 )
   {
      str << array.getElement( 0 );
      for( Index i = 1; i < array.getSize(); i++ )
         str << ", " << array.getElement( i );
   }
   str << " ]";
   return str;
}

// Serialization of arrays into binary files.
template< typename Value, typename Device, typename Index >
File& operator<<( File& file, const Array< Value, Device, Index >& array )
{
   saveObjectType( file, array.getSerializationType() );
   const Index size = array.getSize();
   file.save( &size );
   Algorithms::ArrayIO< Value, Device, Index >::save( file, array.getData(), array.getSize() );
   return file;
}

template< typename Value, typename Device, typename Index >
File& operator<<( File&& file, const Array< Value, Device, Index >& array )
{
   File& f = file;
   return f << array;
}

// Deserialization of arrays from binary files.
template< typename Value, typename Device, typename Index >
File& operator>>( File& file, Array< Value, Device, Index >& array )
{
   const String type = getObjectType( file );
   if( type != array.getSerializationType() )
      throw Exceptions::FileDeserializationError( file.getFileName(), "object type does not match (expected " + array.getSerializationType() + ", found " + type + ")." );
   Index _size;
   file.load( &_size );
   if( _size < 0 )
      throw Exceptions::FileDeserializationError( file.getFileName(), "invalid array size: " + std::to_string(_size) );
   array.setSize( _size );
   Algorithms::ArrayIO< Value, Device, Index >::load( file, array.getData(), array.getSize() );
   return file;
}

template< typename Value, typename Device, typename Index >
File& operator>>( File&& file, Array< Value, Device, Index >& array )
{
   File& f = file;
   return f >> array;
}

} // namespace Containers
} // namespace TNL
