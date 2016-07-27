/***************************************************************************
                          tnlConstSharedArray_impl.h  -  description
                             -------------------
    begin                : Feb 13, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>
#include <TNL/File.h>
#include <TNL/Arrays/Array.h>
#include <TNL/Arrays/ArrayOperations.h>
#include <TNL/core/mfuncs.h>
#include <TNL/core/param-types.h>

namespace TNL {
namespace Arrays {   

template< typename Element,
          typename Device,
          typename Index >
tnlConstSharedArray< Element, Device, Index > :: tnlConstSharedArray()
: size( 0 ), data( 0 )
{
};

template< typename Element,
          typename Device,
          typename Index >
String tnlConstSharedArray< Element, Device, Index > :: getType()
{
   return String( "tnlConstSharedArray< " ) + ", " +
                    TNL::getType< Element >() + ", " +
                     Device::getDeviceType() + ", " +
                    TNL::getType< Index >() + " >";
};

template< typename Element,
          typename Device,
          typename Index >
String tnlConstSharedArray< Element, Device, Index > :: getTypeVirtual() const
{
   return this->getType();
};

template< typename Element,
          typename Device,
          typename Index >
String tnlConstSharedArray< Element, Device, Index > :: getSerializationType()
{
   return Array< Element, Device, Index >::getSerializationType();
};

template< typename Element,
          typename Device,
          typename Index >
String tnlConstSharedArray< Element, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Element,
          typename Device,
          typename Index >
void tnlConstSharedArray< Element, Device, Index > :: bind( const Element* data,
                                                            const Index size )
{
   Assert( size >= 0,
              std::cerr << "You try to set size of tnlConstSharedArray to negative value."
                        << "New size: " << size << std::endl );
   Assert( data != 0,
              std::cerr << "You try to use null pointer to data for tnlConstSharedArray." );

   this->size = size;
   this->data = data;
};

template< typename Element,
          typename Device,
          typename Index >
   template< typename Array >
void tnlConstSharedArray< Element, Device, Index > :: bind( const Array& array,
                                                            IndexType index,
                                                            IndexType size )
{
   // TODO: This does not work for static arrays.
   //tnlStaticAssert( Array::DeviceType::DeviceType == DeviceType::DeviceType,
   //                 "Attempt to bind arrays between different devices." );
   this->data = &( array. getData()[ index ] );
   if( ! size )
      this->size = array. getSize();
   else
      this->size = size;
 
};

template< typename Element,
          typename Device,
          typename Index >
void tnlConstSharedArray< Element, Device, Index > :: swap( tnlConstSharedArray< Element, Device, Index >& array )
{
   swap( this->size, array. size );
   swap( this->data, array. data );
};

template< typename Element,
          typename Device,
          typename Index >
void tnlConstSharedArray< Element, Device, Index > :: reset()
{
   this->size = 0;
   this->data = 0;
};

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
Index tnlConstSharedArray< Element, Device, Index > :: getSize() const
{
   return this->size;
}

template< typename Element,
          typename Device,
          typename Index >
Element tnlConstSharedArray< Element, Device, Index > :: getElement( Index i ) const
{
   Assert( 0 <= i && i < this->getSize(),
              std::cerr << "Wrong index for getElement method in tnlConstSharedArray with name "
                        << " index is " << i
                        << " and array size is " << this->getSize() );
   return ArrayOperations< Device >::getMemoryElement( &( this->data[ i ] ) );
};

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
const Element& tnlConstSharedArray< Element, Device, Index > :: operator[] ( Index i ) const
{
   Assert( 0 <= i && i < this->getSize(),
              std::cerr << "Wrong index for operator[] in tnlConstSharedArray with name "
                        << " index is " << i
                        << " and array size is " << this->getSize() );
   // TODO: add static assert - this does not make sense for Devices::CudaDevice
   return ArrayOperations< Device >::getArrayElementReference( this->data, i );
};

template< typename Element,
           typename Device,
           typename Index >
tnlConstSharedArray< Element, Device, Index >&
    tnlConstSharedArray< Element, Device, Index > :: operator = ( const tnlConstSharedArray< Element, Device, Index >& array )
{
   this->bind( array );
   return ( *this );
};

template< typename Element,
          typename Device,
          typename Index >
   template< typename Array >
tnlConstSharedArray< Element, Device, Index >& tnlConstSharedArray< Element, Device, Index > :: operator = ( const Array& array )
{
   this->bind( array );
   return ( *this );
};

template< typename Element,
          typename Device,
          typename Index >
   template< typename Array >
bool tnlConstSharedArray< Element, Device, Index > :: operator == ( const Array& array ) const
{
   if( array. getSize() != this->getSize() )
      return false;
   return ArrayOperations< Device,
                              typename Array :: DeviceType > ::
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
bool tnlConstSharedArray< Element, Device, Index > :: operator != ( const Array& array ) const
{
   return ! ( ( *this ) == array );
}

template< typename Element,
          typename Device,
          typename Index >
const Element* tnlConstSharedArray< Element, Device, Index > :: getData() const
{
   return this->data;
}

template< typename Element,
          typename Device,
          typename Index >
tnlConstSharedArray< Element, Device, Index > :: operator bool() const
{
   return data != 0;
};


template< typename Element,
          typename Device,
          typename Index >
   template< typename IndexType2 >
void tnlConstSharedArray< Element, Device, Index > :: touch( IndexType2 touches ) const
{
   //TODO: implement
};

template< typename Element,
          typename Device,
          typename Index >
bool tnlConstSharedArray< Element, Device, Index > :: save( File& file ) const
{
   Assert( this->size != 0,
              std::cerr << "You try to save empty array." );
   if( ! Object :: save( file ) )
      return false;
#ifdef HAVE_NOT_CXX11
   if( ! file. write< const Index, Device >( &this->size ) )
#else
   if( ! file. write( &this->size ) )
#endif
      return false;
   if( ! file. write< Element, Device, Index >( this->data, this->size ) )
   {
      std::cerr << "I was not able to WRITE tnlConstSharedArray "
                << " with size " << this->getSize() << std::endl;
      return false;
   }
   return true;
};

template< typename Element,
          typename Device,
          typename Index >
bool tnlConstSharedArray< Element, Device, Index > :: save( const String& fileName ) const
{
   return Object :: save( fileName );
};

template< typename Element, typename Device, typename Index >
std::ostream& operator << ( std::ostream& str, const tnlConstSharedArray< Element, Device, Index >& v )
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

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
extern template class tnlConstSharedArray< float, Devices::Host, int >;
#endif
extern template class tnlConstSharedArray< double, Devices::Host, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlConstSharedArray< long double, Devices::Host, int >;
#endif

#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class tnlConstSharedArray< float, Devices::Host, long int >;
#endif
extern template class tnlConstSharedArray< double, Devices::Host, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlConstSharedArray< long double, Devices::Host, long int >;
#endif
#endif

#ifdef HAVE_CUDA
#ifdef INSTANTIATE_FLOAT
extern template class tnlConstSharedArray< float, Devices::Cuda, int >;
#endif
extern template class tnlConstSharedArray< double, Devices::Cuda, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlConstSharedArray< long double, Devices::Cuda, int >;
#endif

#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class tnlConstSharedArray< float, Devices::Cuda, long int >;
#endif
extern template class tnlConstSharedArray< double, Devices::Cuda, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlConstSharedArray< long double, Devices::Cuda, long int >;
#endif

#endif
#endif

#endif

} // namespace Arrays
} // namespace TNL
