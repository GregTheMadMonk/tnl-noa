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
#include <TNL/File.h>
#include <TNL/Math.h>
#include <TNL/param-types.h>
#include <TNL/Containers/Algorithms/ArrayOperations.h>
#include <TNL/Containers/ArrayIO.h>
#include <TNL/Containers/Array.h>

namespace TNL {
namespace Containers {

template< typename Element,
          typename Device,
          typename Index >
Array< Element, Device, Index >::
Array()
: size( 0 ),
  data( 0 ),
  allocationPointer( 0 ),
  referenceCounter( 0 )
{
}

template< typename Element,
          typename Device,
          typename Index >
Array< Element, Device, Index >::
Array( const IndexType& size )
: size( 0 ),
  data( 0 ),
  allocationPointer( 0 ),
  referenceCounter( 0 )
{
   this->setSize( size );
}

template< typename Element,
          typename Device,
          typename Index >
Array< Element, Device, Index >::
Array( Element* data,
       const IndexType& size )
: size( size ),
  data( data ),
  allocationPointer( 0 ),
  referenceCounter( 0 )
{
}

template< typename Element,
          typename Device,
          typename Index >
Array< Element, Device, Index >::
Array( Array< Element, Device, Index >& array,
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

template< typename Element,
          typename Device,
          typename Index >
String
Array< Element, Device, Index >::
getType()
{
   return String( "Containers::Array< " ) +
          TNL::getType< Element >() + ", " +
          Device::getDeviceType() + ", " +
          TNL::getType< Index >() + " >";
}

template< typename Element,
          typename Device,
          typename Index >
String
Array< Element, Device, Index >::
getTypeVirtual() const
{
   return this->getType();
}

template< typename Element,
          typename Device,
          typename Index >
String
Array< Element, Device, Index >::
getSerializationType()
{
   return HostType::getType();
}

template< typename Element,
          typename Device,
          typename Index >
String
Array< Element, Device, Index >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Element,
          typename Device,
          typename Index >
void
Array< Element, Device, Index >::
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

template< typename Element,
          typename Device,
          typename Index >
void
Array< Element, Device, Index >::
setSize( const Index size )
{
   TNL_ASSERT_GE( size, 0, "Array size must be non-negative." );

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

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
Index
Array< Element, Device, Index >::
getSize() const
{
   return this -> size;
}

template< typename Element,
          typename Device,
          typename Index >
   template< typename ArrayT >
void
Array< Element, Device, Index >::
setLike( const ArrayT& array )
{
   setSize( array.getSize() );
}

template< typename Element,
          typename Device,
          typename Index >
void
Array< Element, Device, Index >::
bind( Element* data,
      const Index size )
{
   TNL_ASSERT_TRUE( data, "Null pointer cannot be bound." );
   this->releaseData();
   this->data = data;
   this->size = size;
}

template< typename Element,
          typename Device,
          typename Index >
   template< typename ArrayT >
void
Array< Element, Device, Index >::
bind( const ArrayT& array,
      const IndexType& begin,
      const IndexType& size )
{
   // all template parameters of Array must match, otherwise binding does not make sense
   static_assert( std::is_same< Element, typename ArrayT::ElementType >::value, "ElementType of both arrays must be the same." );
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
   this->data = const_cast< Element* >( &array.getData()[ begin ] );
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

template< typename Element,
          typename Device,
          typename Index >
   template< int Size >
void
Array< Element, Device, Index >::
bind( StaticArray< Size, Element >& array )
{
   this->releaseData();
   this->size = Size;
   this->data = array.getData();
}


template< typename Element,
          typename Device,
          typename Index >
void
Array< Element, Device, Index >::
swap( Array< Element, Device, Index >& array )
{
   TNL::swap( this->size, array.size );
   TNL::swap( this->data, array.data );
   TNL::swap( this->allocationPointer, array.allocationPointer );
   TNL::swap( this->referenceCounter, array.referenceCounter );
}

template< typename Element,
          typename Device,
          typename Index >
void
Array< Element, Device, Index >::
reset()
{
   this->releaseData();
}

template< typename Element,
          typename Device,
          typename Index >
void
Array< Element, Device, Index >::
setElement( const Index& i, const Element& x )
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, this->getSize(), "Element index is out of bounds." );
   return Algorithms::ArrayOperations< Device >::setMemoryElement( &( this->data[ i ] ), x );
}

template< typename Element,
          typename Device,
          typename Index >
Element
Array< Element, Device, Index >::
getElement( const Index& i ) const
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, this->getSize(), "Element index is out of bounds." );
   return Algorithms::ArrayOperations< Device >::getMemoryElement( & ( this->data[ i ] ) );
}

template< typename Element,
          typename Device,
          typename Index >
bool
Array< Element, Device, Index >::
containsValue( const Element& v ) const
{
   return Algorithms::ArrayOperations< Device >::containsValue( this->data, this->size, v );
}

template< typename Element,
          typename Device,
          typename Index >
bool
Array< Element, Device, Index >::
containsOnlyValue( const Element& v ) const
{
   return Algorithms::ArrayOperations< Device >::containsOnlyValue( this->data, this->size, v );
}

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
inline Element&
Array< Element, Device, Index >::
operator[] ( const Index& i )
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, this->getSize(), "Element index is out of bounds." );
   return this->data[ i ];
}

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
inline const Element&
Array< Element, Device, Index >::
operator[] ( const Index& i ) const
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, this->getSize(), "Element index is out of bounds." );
   return this->data[ i ];
}

template< typename Element,
          typename Device,
          typename Index >
Array< Element, Device, Index >&
Array< Element, Device, Index >::
operator = ( const Array< Element, Device, Index >& array )
{
   //TNL_ASSERT_EQ( array.getSize(), this->getSize(), "Array sizes must be the same." );
   if( this->getSize() != array.getSize() )
      this->setLike( array );
   if( this->getSize() > 0 )
      Algorithms::ArrayOperations< Device >::
         copyMemory( this->getData(),
                     array.getData(),
                     array.getSize() );
   return ( *this );
}

template< typename Element,
          typename Device,
          typename Index >
   template< typename ArrayT >
Array< Element, Device, Index >&
Array< Element, Device, Index >::
operator = ( const ArrayT& array )
{
   //TNL_ASSERT_EQ( array.getSize(), this->getSize(), "Array sizes must be the same." );
   if( this->getSize() != array.getSize() )
      this->setLike( array );   
   if( this->getSize() > 0 )
      Algorithms::ArrayOperations< Device, typename ArrayT::DeviceType >::
         copyMemory( this->getData(),
                     array.getData(),
                     array.getSize() );
   return ( *this );
}

template< typename Element,
          typename Device,
          typename Index >
   template< typename ArrayT >
bool
Array< Element, Device, Index >::
operator == ( const ArrayT& array ) const
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

template< typename Element,
          typename Device,
          typename Index >
   template< typename ArrayT >
bool Array< Element, Device, Index >::operator != ( const ArrayT& array ) const
{
   return ! ( ( *this ) == array );
}


template< typename Element,
          typename Device,
          typename Index >
void Array< Element, Device, Index >::setValue( const Element& e )
{
   TNL_ASSERT_TRUE( this->getData(), "Attempted to set a value of an empty array." );
   Algorithms::ArrayOperations< Device >::setMemory( this->getData(), e, this->getSize() );
}

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
const Element* Array< Element, Device, Index >::getData() const
{
   return this -> data;
}

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
Element* Array< Element, Device, Index >::getData()
{
   return this -> data;
}

template< typename Element,
          typename Device,
          typename Index >
Array< Element, Device, Index >::operator bool() const
{
   return data != 0;
}


template< typename Element,
          typename Device,
          typename Index >
   template< typename IndexType2 >
void Array< Element, Device, Index >::touch( IndexType2 touches ) const
{
   //TODO: implement
}

template< typename Element,
          typename Device,
          typename Index >
bool Array< Element, Device, Index >::save( File& file ) const
{
   if( ! Object::save( file ) )
      return false;
   if( ! file.write( &this->size ) )
      return false;
   if( this->size != 0 && ! ArrayIO< Element, Device, Index >::save( file, this->data, this->size ) )
   {
      std::cerr << "I was not able to save " << this->getType()
           << " with size " << this -> getSize() << std::endl;
      return false;
   }
   return true;
}

template< typename Element,
          typename Device,
          typename Index >
bool
Array< Element, Device, Index >::
load( File& file )
{
   if( ! Object::load( file ) )
      return false;
   Index _size;
   if( ! file.read( &_size ) )
   {
      std::cerr << "Unable to read the array size." << std::endl;
      return false;
   }
   if( _size < 0 )
   {
      std::cerr << "Error: The size " << _size << " of the file is not a positive number or zero." << std::endl;
      return false;
   }
   setSize( _size );
   if( _size )
   {
      if( ! ArrayIO< Element, Device, Index >::load( file, this->data, this->size ) )
      {
         std::cerr << "I was not able to load " << this->getType()
                    << " with size " << this -> getSize() << std::endl;
         return false;
      }
   }
   return true;
}

template< typename Element,
          typename Device,
          typename Index >
bool
Array< Element, Device, Index >::
boundLoad( File& file )
{
   if( ! Object::load( file ) )
      return false;
   Index _size;
   if( ! file.read( &_size ) )
      return false;
   if( _size < 0 )
   {
      std::cerr << "Error: The size " << _size << " of the file is not a positive number or zero." << std::endl;
      return false;
   }
   if( this->getSize() != 0 )
   {
      if( this->getSize() != _size )
      {
         std::cerr << "Error: The current array size is not zero ( " << this->getSize() << ") and it is different from the size of" << std::endl
                   << "the array being loaded (" << _size << "). This is not possible. Call method reset() before." << std::endl;
         return false;
      }
   }
   else setSize( _size );
   if( _size )
   {
      if( ! ArrayIO< Element, Device, Index >::load( file, this->data, this->size ) )
      {
         std::cerr << "I was not able to load " << this->getType()
                    << " with size " << this -> getSize() << std::endl;
         return false;
      }
   }
   return true;
}

template< typename Element,
          typename Device,
          typename Index >
Array< Element, Device, Index >::
~Array()
{
   this->releaseData();
}

template< typename Element, typename Device, typename Index >
std::ostream& operator << ( std::ostream& str, const Array< Element, Device, Index >& v )
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


#ifdef UNDEF //TEMPLATE_EXPLICIT_INSTANTIATION

// TODO: this does not work with CUDA 5.5 - fix it later

#ifdef INSTANTIATE_FLOAT
extern template class Array< float, Devices::Host, int >;
#endif
extern template class Array< double, Devices::Host, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class Array< long double, Devices::Host, int >;
#endif

#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class Array< float, Devices::Host, long int >;
#endif
extern template class Array< double, Devices::Host, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class Array< long double, Devices::Host, long int >;
#endif
#endif

#ifdef HAVE_CUDA
/*
 #ifdef INSTANTIATE_FLOAT
 extern template class Array< float, Devices::Cuda, int >;
 #endif
 extern template class Array< double, Devices::Cuda, int >;
 #ifdef INSTANTIATE_FLOAT
 extern template class Array< float, Devices::Cuda, long int >;
 #endif
 extern template class Array< double, Devices::Cuda, long int >;*/
#endif

#endif

} // namespace Containers
} // namespace TNL
