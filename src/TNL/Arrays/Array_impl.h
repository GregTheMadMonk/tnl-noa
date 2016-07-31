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
#include <TNL/core/mfuncs.h>
#include <TNL/core/param-types.h>
#include <TNL/Arrays/ArrayOperations.h>
#include <TNL/Arrays/ArrayIO.h>
#include <TNL/Arrays/Array.h>

namespace TNL {
namespace Arrays {   

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
};

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
   Assert( begin < array.getSize(),
              std::cerr << " begin = " << begin << " array.getSize() = " << array.getSize() );
   Assert( begin + size  < array.getSize(),
              std::cerr << " begin = " << begin << " size = " << size <<  " array.getSize() = " << array.getSize() );
   if( ! this->size )
      this->size = array.getSize() - begin;
   if( array.allocationPointer )
   {
      if( array.referenceCounter )
      {
         this->referenceCounter = array.referenceCounter;
         *this->referenceCounter++;
      }
      else
      {
         this->referenceCounter = array.referenceCounter = new int;
         *this->referenceCounter = 2;
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
   return String( "Array< " ) +
                    TNL::getType< Element >() + ", " +
                     Device :: getDeviceType() + ", " +
                    TNL::getType< Index >() +
                     " >";
};

template< typename Element,
           typename Device,
           typename Index >
String
Array< Element, Device, Index >::
getTypeVirtual() const
{
   return this->getType();
};

template< typename Element,
           typename Device,
           typename Index >
String
Array< Element, Device, Index >::
getSerializationType()
{
   return HostType::getType();
};

template< typename Element,
           typename Device,
           typename Index >
String
Array< Element, Device, Index >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

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
         ArrayOperations< Device >::freeMemory( this->allocationPointer );
         delete this->referenceCounter;
         //std::cerr << "Deallocating reference counter " << this->referenceCounter << std::endl;
      }
   }
   else
      if( allocationPointer )
         ArrayOperations< Device >::freeMemory( this->allocationPointer );
   this->allocationPointer = 0;
   this->data = 0;
   this->size = 0;
   this->referenceCounter = 0;
}

template< typename Element,
          typename Device,
          typename Index >
bool
Array< Element, Device, Index >::
setSize( const Index size )
{
   Assert( size >= 0,
              std::cerr << "You try to set size of Array to negative value."
                        << "New size: " << size << std::endl );
   if( this->size == size && allocationPointer && ! referenceCounter ) return true;
   this->releaseData();
   ArrayOperations< Device >::allocateMemory( this->allocationPointer, size );
   this->data = this->allocationPointer;
   this->size = size;
   if( ! this->allocationPointer )
   {
      std::cerr << "I am not able to allocate new array with size "
                << ( double ) this->size * sizeof( ElementType ) / 1.0e9 << " GB." << std::endl;
      this->size = 0;
      return false;
   }
   return true;
};

template< typename Element,
           typename Device,
           typename Index >
   template< typename ArrayT >
bool
Array< Element, Device, Index >::
setLike( const ArrayT& array )
{
   Assert( array. getSize() >= 0,
              std::cerr << "You try to set size of Array to negative value."
                        << "Array size: " << array. getSize() << std::endl );
   return setSize( array.getSize() );
};

template< typename Element,
           typename Device,
           typename Index >
void
Array< Element, Device, Index >::
bind( Element* data,
      const Index size )
{
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
   Assert( ( std::is_same< Device, typename ArrayT::DeviceType>::value ), );
   Assert( begin <= array.getSize(),
              std::cerr << " begin = " << begin << " array.getSize() = " << array.getSize() );
   Assert( begin + size  <= array.getSize(),
              std::cerr << " begin = " << begin << " size = " << size <<  " array.getSize() = " << array.getSize() );
 
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
         this->referenceCounter = array.referenceCounter = new int;
         *this->referenceCounter = 2;
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
bind( tnlStaticArray< Size, Element >& array )
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
};

template< typename Element,
          typename Device,
          typename Index >
void
Array< Element, Device, Index >::
reset()
{
   this->releaseData();
};

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
Index
Array< Element, Device, Index >::
getSize() const
{
   return this->size;
}

template< typename Element,
           typename Device,
           typename Index >
void
Array< Element, Device, Index >::
setElement( const Index& i, const Element& x )
{
   Assert( 0 <= i && i < this->getSize(),
              std::cerr << "Wrong index for setElement method in Array "
                        << " index is " << i
                        << " and array size is " << this->getSize() );
   return ArrayOperations< Device > :: setMemoryElement( &( this->data[ i ] ), x );
};

template< typename Element,
           typename Device,
           typename Index >
Element
Array< Element, Device, Index >::
getElement( const Index& i ) const
{
   Assert( 0 <= i && i < this->getSize(),
              std::cerr << "Wrong index for getElement method in Array "
                        << " index is " << i
                        << " and array size is " << this->getSize() );
   return ArrayOperations< Device > :: getMemoryElement( & ( this->data[ i ] ) );
};

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
inline Element&
Array< Element, Device, Index >::
operator[] ( const Index& i )
{
   Assert( 0 <= i && i < this->getSize(),
              std::cerr << "Wrong index for operator[] in Array "
                        << " index is " << i
                        << " and array size is " << this->getSize() );
   return this->data[ i ];
};

template< typename Element,
           typename Device,
           typename Index >
__cuda_callable__
inline const Element&
Array< Element, Device, Index >::
operator[] ( const Index& i ) const
{
   Assert( 0 <= i && i < this->getSize(),
              std::cerr << "Wrong index for operator[] in Array "
                        << " index is " << i
                        << " and array size is " << this->getSize() );
   return this->data[ i ];
};

template< typename Element,
           typename Device,
           typename Index >
Array< Element, Device, Index >&
Array< Element, Device, Index >::
operator = ( const Array< Element, Device, Index >& array )
{
   Assert( array. getSize() == this->getSize(),
              std::cerr << "Source size: " << array. getSize() << std::endl
                        << "Target size: " << this->getSize() << std::endl );
   ArrayOperations< Device > ::
   template copyMemory< Element,
                        Element,
                        Index >
                       ( this->getData(),
                         array. getData(),
                         array. getSize() );
   return ( *this );
};

template< typename Element,
           typename Device,
           typename Index >
   template< typename ArrayT >
Array< Element, Device, Index >&
Array< Element, Device, Index >::
operator = ( const ArrayT& array )
{
   Assert( array. getSize() == this->getSize(),
              std::cerr << "Source size: " << array. getSize() << std::endl
                        << "Target size: " << this->getSize() << std::endl );
   ArrayOperations< Device, typename ArrayT::DeviceType > ::
    template copyMemory< Element,
                         typename ArrayT::ElementType,
                         typename ArrayT::IndexType >
                       ( this->getData(),
                         array. getData(),
                         array. getSize() );
   return ( *this );
};

template< typename Element,
          typename Device,
          typename Index >
   template< typename ArrayT >
bool
Array< Element, Device, Index >::
operator == ( const ArrayT& array ) const
{
   if( array. getSize() != this->getSize() )
      return false;
   return ArrayOperations< Device, typename ArrayT::DeviceType > ::
    template compareMemory< typename ArrayT::ElementType,
                            Element,
                            typename ArrayT::IndexType >
                          ( this->getData(),
                            array.getData(),
                            array.getSize() );
}

template< typename Element,
          typename Device,
          typename Index >
   template< typename ArrayT >
bool Array< Element, Device, Index > :: operator != ( const ArrayT& array ) const
{
   return ! ( ( *this ) == array );
}


template< typename Element,
          typename Device,
          typename Index >
void Array< Element, Device, Index > :: setValue( const Element& e )
{
   Assert( this->getData(),);
   ArrayOperations< Device > :: setMemory( this->getData(), e, this->getSize() );
}

template< typename Element,
           typename Device,
           typename Index >
__cuda_callable__
const Element* Array< Element, Device, Index > :: getData() const
{
   return this->data;
}

template< typename Element,
           typename Device,
           typename Index >
__cuda_callable__
Element* Array< Element, Device, Index > :: getData()
{
   return this->data;
}

template< typename Element,
           typename Device,
           typename Index >
Array< Element, Device, Index > :: operator bool() const
{
   return data != 0;
};


template< typename Element,
           typename Device,
           typename Index >
   template< typename IndexType2 >
void Array< Element, Device, Index > :: touch( IndexType2 touches ) const
{
   //TODO: implement
};

template< typename Element,
          typename Device,
          typename Index >
bool Array< Element, Device, Index > :: save( File& file ) const
{
   if( ! Object :: save( file ) )
      return false;
#ifdef HAVE_NOT_CXX11
   if( ! file. write< const Index, Devices::Host >( &this->size ) )
      return false;
#else
   if( ! file. write( &this->size ) )
      return false;
#endif
   if( this->size != 0 && ! ArrayIO< Element, Device, Index >::save( file, this->data, this->size ) )
   {
      std::cerr << "I was not able to save " << this->getType()
                << " with size " << this->getSize() << std::endl;
      return false;
   }
   return true;
};

template< typename Element,
          typename Device,
          typename Index >
bool
Array< Element, Device, Index >::
load( File& file )
{
   if( ! Object :: load( file ) )
      return false;
   Index _size;
#ifdef HAVE_NOT_CXX11
   if( ! file. read< Index, Devices::Host >( &_size ) )
      return false;
#else
   if( ! file. read( &_size ) )
      return false;
#endif
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
                   << " with size " << this->getSize() << std::endl;
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
   if( ! Object :: load( file ) )
      return false;
   Index _size;
#ifdef HAVE_NOT_CXX11
   if( ! file. read< Index, Devices::Host >( &_size ) )
      return false;
#else
   if( ! file. read( &_size ) )
      return false;
#endif
   if( _size < 0 )
   {
      std::cerr << "Error: The size " << _size << " of the file is not a positive number or zero." << std::endl;
      return false;
   }
   if( this->getSize() != 0 )
   {
      if( this->getSize() != _size )
      {
         std::cerr << "Error: The current array size is not zero and it is different from the size of" << std::endl
                   << "the array being loaded. This is not possible. Call method reset() before." << std::endl;
         return false;
      }
   }
   else setSize( _size );
   if( _size )
   {
      if( ! ArrayIO< Element, Device, Index >::load( file, this->data, this->size ) )
      {
         std::cerr << "I was not able to load " << this->getType()
                   << " with size " << this->getSize() << std::endl;
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
boundLoad( const String& fileName )
{
   File file;
   if( ! file. open( fileName, tnlReadMode ) )
   {
      std::cerr << "I am not bale to open the file " << fileName << " for reading." << std::endl;
      return false;
   }
   if( ! this->boundLoad( file ) )
      return false;
   if( ! file. close() )
   {
      std::cerr << "An error occurred when I was closing the file " << fileName << "." << std::endl;
      return false;
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
         str << ", " << v. getElement( i );
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

} // namespace Arrays
} // namespace TNL
