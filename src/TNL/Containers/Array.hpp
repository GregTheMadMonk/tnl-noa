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

#include <TNL/Assert.h>
#include <TNL/Math.h>
#include <TNL/param-types.h>
#include <TNL/Containers/Algorithms/ArrayOperations.h>
#include <TNL/Containers/Algorithms/ArrayIO.h>
#include <TNL/Containers/Algorithms/ArrayAssignment.h>
#include <TNL/Exceptions/ArrayWrongSize.h>

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
: size( size ),
  data( data ),
  allocationPointer( 0 ),
  referenceCounter( 0 )
{
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
Array( Array< Value, Device, Index >& array,
       const IndexType& begin,
       const IndexType& size )
: size( size ),
  data( &array.getData()[ begin ] ),
  allocationPointer( array.allocationPointer ),
  referenceCounter( 0 )
{
   TNL_ASSERT_TRUE( array.getData(), "Empty arrays cannot be bound." );
   TNL_ASSERT_LT( begin, array.getSize(), "Begin of array is out of bounds." );
   TNL_ASSERT_LE( begin + size, array.getSize(), "End of array is out of bounds." );

   if( ! this->size )
      this->size = array.getSize() - begin;
   if( array.allocationPointer )
   {
      if( array.referenceCounter )
      {
         this->referenceCounter = array.referenceCounter;
         *this->referenceCounter += 1;
      }
      else
      {
         this->referenceCounter = array.referenceCounter = new int( 2 );
      }
   }
}

template< typename Value,
          typename Device,
          typename Index >
Array< Value, Device, Index >::
Array( Array< Value, Device, Index >&& array )
{
   this->size = array.size;
   this->data = array.data;
   this->allocationPointer = array.allocationPointer;
   this->referenceCounter = array.referenceCounter;

   array.size = 0;
   array.data = nullptr;
   array.allocationPointer = nullptr;
   array.referenceCounter = nullptr;
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
   ////
   // Here we assume that the underlying array for initializer_list is const T[N]
   // as noted here:
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
setSize( const Index size )
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
getView()
{
   return ViewType( getData(), getSize() );
}

template< typename Value,
          typename Device,
          typename Index >
typename Array< Value, Device, Index >::ConstViewType
Array< Value, Device, Index >::
getConstView() const
{
   return ConstViewType( getData(), getSize() );
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
const Value* Array< Value, Device, Index >::getData() const
{
   return this->data;
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
Value* Array< Value, Device, Index >::getData()
{
   return this->data;
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
const Value* Array< Value, Device, Index >::getArrayData() const
{
   return this->data;
}

template< typename Value,
          typename Device,
          typename Index >
__cuda_callable__
Value* Array< Value, Device, Index >::getArrayData()
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
void Array< Value, Device, Index >::setValue( const ValueType& e,
                                              const Index begin,
                                              Index end )
{
   TNL_ASSERT_TRUE( this->getData(), "Attempted to set a value of an empty array." );
   if( end == -1 )
      end = this->getSize();
   Algorithms::ArrayOperations< Device >::setMemory( &this->getData()[ begin ], e, end - begin );
}

template< typename Value,
          typename Device,
          typename Index >
bool
Array< Value, Device, Index >::
containsValue( const Value& v,
               const Index begin,
               Index end ) const
{
   TNL_ASSERT_TRUE( this->getData(), "Attempted to check a value of an empty array." );
   if( end == -1 )
      end = this->getSize();

   return Algorithms::ArrayOperations< Device >::containsValue( &this->getData()[ begin ], end - begin, v );
}

template< typename Value,
          typename Device,
          typename Index >
bool
Array< Value, Device, Index >::
containsOnlyValue( const Value& v,
                   const Index begin,
                   Index end ) const
{
   TNL_ASSERT_TRUE( this->getData(), "Attempted to check a value of an empty array." );
   if( end == -1 )
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
void Array< Value, Device, Index >::save( File& file ) const
{
   Object::save( file );
   file.save( &this->size );
   if( this->size != 0 )
      Algorithms::ArrayIO< Value, Device, Index >::save( file, this->data, this->size );
}

template< typename Value,
          typename Device,
          typename Index >
void
Array< Value, Device, Index >::
load( File& file )
{
   Object::load( file );
   Index _size;
   file.load( &_size );
   if( _size < 0 )
      throw Exceptions::ArrayWrongSize( _size, "positive" );
   setSize( _size );
   if( _size )
      Algorithms::ArrayIO< Value, Device, Index >::load( file, this->data, this->size );
}

template< typename Value,
          typename Device,
          typename Index >
void
Array< Value, Device, Index >::
boundLoad( File& file )
{
   Object::load( file );
   Index _size;
   file.load( &_size );
   if( _size < 0 )
      throw Exceptions::ArrayWrongSize( _size, "Positive is expected," );
   if( this->getSize() != 0 )
   {
      if( this->getSize() != _size )
         throw Exceptions::ArrayWrongSize( _size, convertToString( this->getSize() ) + " is expected." );
   }
   else setSize( _size );
   if( _size )
      Algorithms::ArrayIO< Value, Device, Index >::load( file, this->data, this->size );
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
std::ostream& operator<<( std::ostream& str, const Array< Value, Device, Index >& v )
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
