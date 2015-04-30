/***************************************************************************
                          tnlSharedArray.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
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

#ifndef TNLSHAREDARRAY_H_IMPLEMENTATION
#define TNLSHAREDARRAY_H_IMPLEMENTATION

#include <iostream>
#include <core/tnlFile.h>
#include <core/arrays/tnlArray.h>
#include <core/arrays/tnlStaticArray.h>
#include <core/arrays/tnlArrayOperations.h>
#include <core/mfuncs.h>
#include <core/param-types.h>

using namespace std;

template< typename Element,
          typename Device,
          typename Index >
tnlSharedArray< Element, Device, Index >::tnlSharedArray()
: size( 0 ), data( 0 )
{
};

template< typename Element,
          typename Device,
          typename Index >
tnlSharedArray< Element, Device, Index >::tnlSharedArray( Element* _data,
                                                          const Index _size )
{
   this->bind( _data, _size );
}

template< typename Element,
          typename Device,
          typename Index >
tnlSharedArray< Element, Device, Index >::tnlSharedArray( tnlArray< Element, Device, Index >& array )
{
   this->bind( array );
}

template< typename Element,
          typename Device,
          typename Index >
tnlSharedArray< Element, Device, Index >::tnlSharedArray( tnlSharedArray< Element, Device, Index >& array )
{
   this->bind( array );
}

template< typename Element,
          typename Device,
          typename Index >
tnlString tnlSharedArray< Element, Device, Index > :: getType()
{
   return tnlString( "tnlSharedArray< " ) + ", " +
                     ::getType< Element >() + ", " +
                     Device::getDeviceType() + ", " +
                     ::getType< Index >() + " >";
};

template< typename Element,
           typename Device,
           typename Index >
tnlString tnlSharedArray< Element, Device, Index > :: getTypeVirtual() const
{
   return this->getType();
};

template< typename Element,
           typename Device,
           typename Index >
tnlString tnlSharedArray< Element, Device, Index > :: getSerializationType()
{
   return tnlArray< Element, Device, Index >::getSerializationType();
};

template< typename Element,
           typename Device,
           typename Index >
tnlString tnlSharedArray< Element, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Element,
          typename Device,
          typename Index >
void tnlSharedArray< Element, Device, Index > :: bind( Element* data,
                                                       const Index size )
{
   tnlAssert( size >= 0,
              cerr << "You try to set size of tnlSharedArray to negative value."
                   << "Name: " << this -> getName() << endl
                   << "New size: " << size << endl );
   tnlAssert( data != 0,
              cerr << "You try to use null pointer to data for tnlSharedArray."
                   << "Name: " << this -> getName() );

   this -> size = size;
   this -> data = data;
};

template< typename Element,
          typename Device,
          typename Index >
void tnlSharedArray< Element, Device, Index > :: bind( tnlArray< Element, Device, Index >& array )
{
   this->size = array. getSize();
   this->data = array. getData();
};

template< typename Element,
          typename Device,
          typename Index >
   template< int Size >
void tnlSharedArray< Element, Device, Index >::bind( tnlStaticArray< Size, Element >& array )
{
   this->size = Size;
   this->data = array.getData();
}

template< typename Element,
          typename Device,
          typename Index >
void tnlSharedArray< Element, Device, Index > :: bind( tnlSharedArray< Element, Device, Index >& array )
{
   this -> size = array. getSize();
   this -> data = array. getData();
};

template< typename Element,
          typename Device,
          typename Index >
void tnlSharedArray< Element, Device, Index > :: swap( tnlSharedArray< Element, Device, Index >& array )
{
   :: swap( this -> size, array. size );
   :: swap( this -> data, array. data );
};

template< typename Element,
          typename Device,
          typename Index >
void tnlSharedArray< Element, Device, Index > :: reset()
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
Index tnlSharedArray< Element, Device, Index > :: getSize() const
{
   return this -> size;
}

template< typename Element,
          typename Device,
          typename Index >
void tnlSharedArray< Element, Device, Index > :: setElement( const Index i, const Element& x )
{
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for setElement method in tnlSharedArray with name "
                   << this -> getName()
                   << " index is " << i
                   << " and array size is " << this -> getSize() );
   return tnlArrayOperations< Device >::setMemoryElement( & ( this -> data[ i ] ), x );
};

template< typename Element,
          typename Device,
          typename Index >
Element tnlSharedArray< Element, Device, Index > :: getElement( Index i ) const
{
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for getElement method in tnlSharedArray with name "
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
Element& tnlSharedArray< Element, Device, Index > :: operator[] ( Index i )
{
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for operator[] in tnlSharedArray with name "
                   << this -> getName()
                   << " index is " << i
                   << " and array size is " << this -> getSize() );
   return this->data[ i ];
};

template< typename Element,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
const Element& tnlSharedArray< Element, Device, Index > :: operator[] ( Index i ) const
{
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for operator[] in tnlSharedArray with name "
                   << this -> getName()
                   << " index is " << i
                   << " and array size is " << this -> getSize() );
   return this->data[ i ];
};

template< typename Element,
           typename Device,
           typename Index >
tnlSharedArray< Element, Device, Index >&
    tnlSharedArray< Element, Device, Index > :: operator = ( const tnlSharedArray< Element, Device, Index >& array )
{
   tnlAssert( array. getSize() == this -> getSize(),
           cerr << "Source name: " << array. getName() << endl
                << "Source size: " << array. getSize() << endl
                << "Target name: " << this -> getName() << endl
                << "Target size: " << this -> getSize() << endl );
   tnlArrayOperations< Device > ::
   template copyMemory< Element,
                        Element,
                        Index >
                       ( this -> getData(),
                         array. getData(),
                         array. getSize() );
   return ( *this );
};

template< typename Element,
          typename Device,
          typename Index >
   template< typename Array >
tnlSharedArray< Element, Device, Index >& tnlSharedArray< Element, Device, Index > :: operator = ( const Array& array )
{
   tnlAssert( array. getSize() == this -> getSize(),
           cerr << "Source name: " << array. getName() << endl
                << "Source size: " << array. getSize() << endl
                << "Target name: " << this -> getName() << endl
                << "Target size: " << this -> getSize() << endl );
   tnlArrayOperations< typename Array :: DeviceType,
                       Device > ::
    template copyMemory< Element,
                         typename Array :: ElementType,
                         typename Array :: IndexType >
                       ( this -> getData(),
                         array. getData(),
                         array. getSize() );
   return ( *this );
};

template< typename Element,
          typename Device,
          typename Index >
   template< typename Array >
bool tnlSharedArray< Element, Device, Index > :: operator == ( const Array& array ) const
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
bool tnlSharedArray< Element, Device, Index > :: operator != ( const Array& array ) const
{
   return ! ( ( *this ) == array );
}

template< typename Element,
          typename Device,
          typename Index >
void tnlSharedArray< Element, Device, Index > :: setValue( const Element& e )
{
   tnlAssert( this -> size != 0,
              cerr << "Array name is " << this -> getName() );
   tnlArrayOperations< Device >::template setMemory< Element, Index >
                              ( this -> getData(), e, this -> getSize() );

}

template< typename Element,
          typename Device,
          typename Index >
const Element* tnlSharedArray< Element, Device, Index > :: getData() const
{
   return this -> data;
}

template< typename Element,
          typename Device,
          typename Index >
Element* tnlSharedArray< Element, Device, Index > :: getData()
{
   return this -> data;
}

template< typename Element,
          typename Device,
          typename Index >
tnlSharedArray< Element, Device, Index > :: operator bool() const
{
   return data != 0;
};


template< typename Element,
          typename Device,
          typename Index >
   template< typename IndexType2 >
void tnlSharedArray< Element, Device, Index > :: touch( IndexType2 touches ) const
{
   //TODO: implement
};

template< typename Element,
          typename Device,
          typename Index >
bool tnlSharedArray< Element, Device, Index > :: save( tnlFile& file ) const
{
   tnlAssert( this -> size != 0,
              cerr << "You try to save empty array. Its name is " << this -> getName() );
   if( ! tnlObject :: save( file ) )
      return false;
#ifdef HAVE_NOT_CXX11
   if( ! file. write< const Index, tnlHost >( &this -> size ) )
#else            
   if( ! file. write( &this -> size ) )
#endif      
      return false;
   if( ! file. write< Element, Device, Index >( this -> data, this -> size ) )
   {
      cerr << "I was not able to WRITE tnlSharedArray " << this -> getName()
           << " with size " << this -> getSize() << endl;
      return false;
   }
   return true;
};

template< typename Element,
          typename Device,
          typename Index >
bool tnlSharedArray< Element, Device, Index > :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
};

template< typename Element,
          typename Device,
          typename Index >
bool tnlSharedArray< Element, Device, Index > :: load( tnlFile& file )
{
   if( ! tnlObject :: load( file ) )
      return false;
   Index _size;
#ifdef HAVE_NOT_CXX11
   if( ! file. read< Index, tnlHost >( &_size ) )
      return false;
#else
   if( ! file. read( &_size, 1 ) )
      return false;
#endif
   if( _size != this->size )
   {
      cerr << "Error: The size " << _size << " of the data to be load is different from the " <<
               "allocated array. This is not possible in the shared array ( in " << this->getName() <<
               " )." << endl;
      return false;
   }
   if( _size )
   {
      if( ! file. read< Element, Device, Index >( this -> data, this -> size ) )
      {
         cerr << "I was not able to READ tnlSharedArray " << this -> getName()
              << " with size " << this -> getSize() << endl;
         return false;
      }
   }
   return true;
};

template< typename Element,
          typename Device,
          typename Index >
bool tnlSharedArray< Element, Device, Index > :: load( const tnlString& fileName )
{
   return tnlObject :: load( fileName );
};


template< typename Element, typename Device, typename Index >
ostream& operator << ( ostream& str, const tnlSharedArray< Element, Device, Index >& v )
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

// TODO: this does not work with CUDA 5.5 - fix it later

#ifdef INSTANTIATE_FLOAT
extern template class tnlSharedArray< float, tnlHost, int >;
#endif
extern template class tnlSharedArray< double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlSharedArray< long double, tnlHost, int >;
#endif

#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class tnlSharedArray< float, tnlHost, long int >;
#endif
extern template class tnlSharedArray< double, tnlHost, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlSharedArray< long double, tnlHost, long int >;
#endif
#endif


#ifdef HAVE_CUDA
/*
#ifdef INSTANTIATE_FLOAT
extern template class tnlSharedArray< float, tnlCuda, int >;
#endif
extern template class tnlSharedArray< double, tnlCuda, int >;
#ifdef INSTANTIATE_FLOAT
extern template class tnlSharedArray< float, tnlCuda, long int >;
#endif
extern template class tnlSharedArray< double, tnlCuda, long int >;*/
#endif

#endif

#endif /* TNLSHAREDARRAY_H_IMPLEMENTATION */
