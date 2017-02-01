/***************************************************************************
                          SharedArray.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
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

namespace TNL {
namespace Containers {   

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
SharedArray< Element, Device, Index >::SharedArray()
: size( 0 ), data( 0 )
{
};

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
SharedArray< Element, Device, Index >::SharedArray( Element* _data,
                                                          const Index _size )
{
   this->bind( _data, _size );
}

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
SharedArray< Element, Device, Index >::SharedArray( Array< Element, Device, Index >& array )
{
   this->bind( array );
}

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
SharedArray< Element, Device, Index >::SharedArray( SharedArray< Element, Device, Index >& array )
{
   this->bind( array );
}

template< typename Element,
          typename Device,
          typename Index >
String SharedArray< Element, Device, Index > :: getType()
{
   return String( "Containers::SharedArray< " ) + ", " +
                    TNL::getType< Element >() + ", " +
                     Device::getDeviceType() + ", " +
                    TNL::getType< Index >() + " >";
};

template< typename Element,
           typename Device,
           typename Index >
String SharedArray< Element, Device, Index > :: getTypeVirtual() const
{
   return this->getType();
};

template< typename Element,
           typename Device,
           typename Index >
String
SharedArray< Element, Device, Index >::
getSerializationType()
{
   return HostType::getType();
};

template< typename Element,
           typename Device,
           typename Index >
String
SharedArray< Element, Device, Index >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
void SharedArray< Element, Device, Index > :: bind( Element* data,
                                                       const Index size )
{
   TNL_ASSERT( size >= 0,
              std::cerr << "You try to set size of SharedArray to negative value."
                        << "New size: " << size << std::endl );
   TNL_ASSERT( data != 0,
              std::cerr << "You try to use null pointer to data for SharedArray." );

   this->size = size;
   this->data = data;
};

template< typename Element,
          typename Device,
          typename Index >
   template< typename Array >
__cuda_callable__
void SharedArray< Element, Device, Index > :: bind( Array& array,
                                                       IndexType index,
                                                       IndexType size )
{
   //tnlStaticTNL_ASSERT( Array::DeviceType::DeviceType == DeviceType::DeviceType,
   //                 "Attempt to bind arrays between different devices." );
   // TODO: fix this - it does nto work with StaticArray
   this->data = &( array. getData()[ index ] );
   if( ! size )
      this->size = array. getSize();
   else
      this->size = size;
 
};

template< typename Element,
          typename Device,
          typename Index >
   template< int Size >
__cuda_callable__
void SharedArray< Element, Device, Index >::bind( StaticArray< Size, Element >& array )
{
   this->size = Size;
   this->data = array.getData();
}

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
void SharedArray< Element, Device, Index > :: bind( SharedArray< Element, Device, Index >& array )
{
   this->size = array. getSize();
   this->data = array. getData();
};

template< typename Element,
          typename Device,
          typename Index >
void SharedArray< Element, Device, Index > :: swap( SharedArray< Element, Device, Index >& array )
{
   TNL::swap( this->size, array. size );
   TNL::swap( this->data, array. data );
};

template< typename Element,
          typename Device,
          typename Index >
void SharedArray< Element, Device, Index > :: reset()
{
   this->size = 0;
   this->data = 0;
};

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
Index SharedArray< Element, Device, Index > :: getSize() const
{
   return this->size;
}

template< typename Element,
          typename Device,
          typename Index >
void SharedArray< Element, Device, Index > :: setElement( const Index& i, const Element& x )
{
   TNL_ASSERT( 0 <= i && i < this->getSize(),
              std::cerr << "Wrong index for setElement method in SharedArray "
                        << " index is " << i
                        << " and array size is " << this->getSize() );
   return Algorithms::ArrayOperations< Device >::setMemoryElement( & ( this->data[ i ] ), x );
};

template< typename Element,
          typename Device,
          typename Index >
Element SharedArray< Element, Device, Index > :: getElement( const Index& i ) const
{
   TNL_ASSERT( 0 <= i && i < this->getSize(),
              std::cerr << "Wrong index for getElement method in SharedArray "
                        << " index is " << i
                        << " and array size is " << this->getSize() );
   return Algorithms::ArrayOperations< Device >::getMemoryElement( &( this->data[ i ] ) );
};

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
Element& SharedArray< Element, Device, Index > :: operator[] ( const Index& i )
{
   TNL_ASSERT( 0 <= i && i < this->getSize(),
              std::cerr << "Wrong index for operator[] in SharedArray "
                        << " index is " << i
                        << " and array size is " << this->getSize() );
   return this->data[ i ];
};

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
const Element& SharedArray< Element, Device, Index > :: operator[] ( const Index& i ) const
{
   TNL_ASSERT( 0 <= i && i < this->getSize(),
              std::cerr << "Wrong index for operator[] in SharedArray "
                        << " index is " << i
                        << " and array size is " << this->getSize() );
   return this->data[ i ];
};

template< typename Element,
           typename Device,
           typename Index >
SharedArray< Element, Device, Index >&
    SharedArray< Element, Device, Index > :: operator = ( const SharedArray< Element, Device, Index >& array )
{
   TNL_ASSERT( array. getSize() == this->getSize(),
              std::cerr << "Source size: " << array. getSize() << std::endl
                        << "Target size: " << this->getSize() << std::endl );
   Algorithms::ArrayOperations< Device > ::
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
   template< typename Array >
SharedArray< Element, Device, Index >& SharedArray< Element, Device, Index > :: operator = ( const Array& array )
{
   TNL_ASSERT( array. getSize() == this->getSize(),
              std::cerr << "Source size: " << array. getSize() << std::endl
                        << "Target size: " << this->getSize() << std::endl );
   Algorithms::ArrayOperations< typename Array::DeviceType, Device >::
      template copyMemory< Element,
                           typename Array :: ElementType,
                           typename Array :: IndexType >
                         ( this->getData(),
                           array. getData(),
                           array. getSize() );
   return ( *this );
};

template< typename Element,
          typename Device,
          typename Index >
   template< typename Array >
bool SharedArray< Element, Device, Index > :: operator == ( const Array& array ) const
{
   if( array. getSize() != this->getSize() )
      return false;
   return Algorithms::ArrayOperations< Device, typename Array::DeviceType >::
      template compareMemory< typename Array :: ElementType,
                              Element,
                              typename Array :: IndexType >
                            ( this->getData(),
                              array. getData(),
                              array. getSize() );
}

template< typename Element,
          typename Device,
          typename Index >
   template< typename Array >
bool SharedArray< Element, Device, Index > :: operator != ( const Array& array ) const
{
   return ! ( ( *this ) == array );
}

template< typename Element,
          typename Device,
          typename Index >
void SharedArray< Element, Device, Index > :: setValue( const Element& e )
{
   TNL_ASSERT( this->size != 0, );
   Algorithms::ArrayOperations< Device >::template setMemory< Element, Index >
                              ( this->getData(), e, this->getSize() );

}

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__ 
const Element* SharedArray< Element, Device, Index > :: getData() const
{
   return this->data;
}

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__ 
Element* SharedArray< Element, Device, Index > :: getData()
{
   return this->data;
}

template< typename Element,
          typename Device,
          typename Index >
SharedArray< Element, Device, Index > :: operator bool() const
{
   return data != 0;
};


template< typename Element,
          typename Device,
          typename Index >
   template< typename IndexType2 >
void SharedArray< Element, Device, Index > :: touch( IndexType2 touches ) const
{
   //TODO: implement
};

template< typename Element,
          typename Device,
          typename Index >
bool SharedArray< Element, Device, Index > :: save( File& file ) const
{
   TNL_ASSERT( this->size != 0,
              std::cerr << "You try to save empty array." << std::endl );
   if( ! Object :: save( file ) )
      return false;
#ifdef HAVE_NOT_CXX11
   if( ! file. write< const Index, Devices::Host >( &this->size ) )
#else
   if( ! file. write( &this->size ) )
#endif
      return false;
   if( ! file. write< Element, Device, Index >( this->data, this->size ) )
   {
      std::cerr << "I was not able to WRITE SharedArray with size " << this->getSize() << std::endl;
      return false;
   }
   return true;
};

template< typename Element,
          typename Device,
          typename Index >
bool SharedArray< Element, Device, Index > :: save( const String& fileName ) const
{
   return Object :: save( fileName );
};

template< typename Element,
          typename Device,
          typename Index >
bool SharedArray< Element, Device, Index > :: load( File& file )
{
   if( ! Object :: load( file ) )
      return false;
   Index _size;
#ifdef HAVE_NOT_CXX11
   if( ! file. read< Index, Devices::Host >( &_size ) )
      return false;
#else
   if( ! file. read( &_size, 1 ) )
      return false;
#endif
   if( _size != this->size )
   {
      std::cerr << "Error: The size " << _size << " of the data to be load is different from the " <<
                   "allocated array. This is not possible in the shared array." << std::endl;
      return false;
   }
   if( _size )
   {
      if( ! file. read< Element, Device, Index >( this->data, this->size ) )
      {
         std::cerr << "I was not able to READ SharedArray with size " << this->getSize() << std::endl;
         return false;
      }
   }
   return true;
};

template< typename Element,
          typename Device,
          typename Index >
bool SharedArray< Element, Device, Index > :: load( const String& fileName )
{
   return Object :: load( fileName );
};


template< typename Element, typename Device, typename Index >
std::ostream& operator << ( std::ostream& str, const SharedArray< Element, Device, Index >& v )
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
extern template class SharedArray< float, Devices::Host, int >;
#endif
extern template class SharedArray< double, Devices::Host, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class SharedArray< long double, Devices::Host, int >;
#endif

#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class SharedArray< float, Devices::Host, long int >;
#endif
extern template class SharedArray< double, Devices::Host, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class SharedArray< long double, Devices::Host, long int >;
#endif
#endif


#ifdef HAVE_CUDA
/*
#ifdef INSTANTIATE_FLOAT
extern template class SharedArray< float, Devices::Cuda, int >;
#endif
extern template class SharedArray< double, Devices::Cuda, int >;
#ifdef INSTANTIATE_FLOAT
extern template class SharedArray< float, Devices::Cuda, long int >;
#endif
extern template class SharedArray< double, Devices::Cuda, long int >;*/
#endif

#endif

} // namespace Containers
} // namespace TNL
