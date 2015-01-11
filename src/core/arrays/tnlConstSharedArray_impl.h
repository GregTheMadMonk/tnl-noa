/***************************************************************************
                          tnlConstSharedArray_impl.h  -  description
                             -------------------
    begin                : Feb 13, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLCONSTSHAREDARRAY_IMPL_H_
#define TNLCONSTSHAREDARRAY_IMPL_H_

#include <iostream>
#include <core/tnlFile.h>
#include <core/arrays/tnlArray.h>
#include <core/arrays/tnlArrayOperations.h>
#include <core/mfuncs.h>
#include <core/param-types.h>

using namespace std;

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
tnlString tnlConstSharedArray< Element, Device, Index > :: getType()
{
   return tnlString( "tnlConstSharedArray< " ) + ", " +
                     ::getType< Element >() + ", " +
                     Device::getDeviceType() + ", " +
                     ::getType< Index >() + " >";
};

template< typename Element,
          typename Device,
          typename Index >
tnlString tnlConstSharedArray< Element, Device, Index > :: getTypeVirtual() const
{
   return this->getType();
};

template< typename Element,
          typename Device,
          typename Index >
tnlString tnlConstSharedArray< Element, Device, Index > :: getSerializationType()
{
   return tnlArray< Element, Device, Index >::getSerializationType();
};

template< typename Element,
          typename Device,
          typename Index >
tnlString tnlConstSharedArray< Element, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Element,
          typename Device,
          typename Index >
void tnlConstSharedArray< Element, Device, Index > :: bind( const Element* data,
                                                            const Index size )
{
   tnlAssert( size >= 0,
              cerr << "You try to set size of tnlConstSharedArray to negative value."
                   << "Name: " << this -> getName() << endl
                   << "New size: " << size << endl );
   tnlAssert( data != 0,
              cerr << "You try to use null pointer to data for tnlConstSharedArray."
                   << "Name: " << this -> getName() );

   this -> size = size;
   this -> data = data;
};

template< typename Element,
          typename Device,
          typename Index >
   template< typename Array >
void tnlConstSharedArray< Element, Device, Index > :: bind( const Array& array )
{
   this -> size = array. getSize();
   this -> data = array. getData();
};

template< typename Element,
          typename Device,
          typename Index >
void tnlConstSharedArray< Element, Device, Index > :: swap( tnlConstSharedArray< Element, Device, Index >& array )
{
   :: swap( this -> size, array. size );
   :: swap( this -> data, array. data );
};

template< typename Element,
          typename Device,
          typename Index >
void tnlConstSharedArray< Element, Device, Index > :: reset()
{
   this -> size = 0;
   this -> data = 0;
};

template< typename Element,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Index tnlConstSharedArray< Element, Device, Index > :: getSize() const
{
   return this -> size;
}

template< typename Element,
          typename Device,
          typename Index >
Element tnlConstSharedArray< Element, Device, Index > :: getElement( Index i ) const
{
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for getElement method in tnlConstSharedArray with name "
                   << this -> getName()
                   << " index is " << i
                   << " and array size is " << this -> getSize() );
   return tnlArrayOperations< Device >::getMemoryElement( &( this -> data[ i ] ) );
};

template< typename Element,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
const Element& tnlConstSharedArray< Element, Device, Index > :: operator[] ( Index i ) const
{
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for operator[] in tnlConstSharedArray with name "
                   << this -> getName()
                   << " index is " << i
                   << " and array size is " << this -> getSize() );
   // TODO: add static assert - this does not make sense for tnlCudaDevice
   return tnlArrayOperations< Device >::getArrayElementReference( this -> data, i );
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
   if( array. getSize() != this -> getSize() )
      return false;
   return tnlArrayOperations< Device,
                              typename Array :: DeviceType > ::
    template compareMemory< typename Array :: ElementType,
                            Element,
                            typename Array :: IndexType >
                          ( this -> getData(),
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
   return this -> data;
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
bool tnlConstSharedArray< Element, Device, Index > :: save( tnlFile& file ) const
{
   tnlAssert( this -> size != 0,
              cerr << "You try to save empty array. Its name is " << this -> getName() );
   if( ! tnlObject :: save( file ) )
      return false;
#ifdef HAVE_NOT_CXX11
   if( ! file. write< const Index, Device >( &this -> size ) )
#else
   if( ! file. write( &this -> size ) )
#endif
      return false;
   if( ! file. write< Element, Device, Index >( this -> data, this -> size ) )
   {
      cerr << "I was not able to WRITE tnlConstSharedArray " << this -> getName()
           << " with size " << this -> getSize() << endl;
      return false;
   }
   return true;
};

template< typename Element,
          typename Device,
          typename Index >
bool tnlConstSharedArray< Element, Device, Index > :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
};

template< typename Element, typename Device, typename Index >
ostream& operator << ( ostream& str, const tnlConstSharedArray< Element, Device, Index >& v )
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

//}; // namespace implementation


#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

extern template class tnlConstSharedArray< float, tnlHost, int >;
extern template class tnlConstSharedArray< double, tnlHost, int >;
extern template class tnlConstSharedArray< float, tnlHost, long int >;
extern template class tnlConstSharedArray< double, tnlHost, long int >;

#ifdef HAVE_CUDA
extern template class tnlConstSharedArray< float, tnlCuda, int >;
extern template class tnlConstSharedArray< double, tnlCuda, int >;
extern template class tnlConstSharedArray< float, tnlCuda, long int >;
extern template class tnlConstSharedArray< double, tnlCuda, long int >;
#endif

#endif



#endif /* TNLCONSTSHAREDARRAY_IMPL_H_ */
