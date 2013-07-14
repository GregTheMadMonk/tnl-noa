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
#include <core/mfuncs.h>
#include <core/param-types.h>

using namespace std;

//namespace implementation
//{

template< typename Element,
          typename Device,
          typename Index >
tnlSharedArray< Element, Device, Index > :: tnlSharedArray()
: size( 0 ), data( 0 )
{
};

template< typename Element,
          typename Device,
          typename Index >
tnlString tnlSharedArray< Element, Device, Index > :: getType() const
{
   return tnlString( "tnlSharedArray< " ) + ", " +
                     getParameterType< Element >() + ", " +
                     Device :: getDeviceType() + ", " +
                     getParameterType< Index >() + " >";
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
   this -> size = array. getSize();
   this -> data = array. getData();
};

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
   return Device :: setMemoryElement( & ( this -> data[ i ] ), x );
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
   return Device :: getMemoryElement( &( this -> data[ i ] ) );
};

template< typename Element,
          typename Device,
          typename Index >
Element& tnlSharedArray< Element, Device, Index > :: operator[] ( Index i )
{
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for operator[] in tnlSharedArray with name "
                   << this -> getName()
                   << " index is " << i
                   << " and array size is " << this -> getSize() );
   // TODO: add static assert - this does not make sense for tnlCudaDevice
   return Device :: getArrayElementReference( this -> data, i );
};

template< typename Element,
          typename Device,
          typename Index >
const Element& tnlSharedArray< Element, Device, Index > :: operator[] ( Index i ) const
{
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for operator[] in tnlSharedArray with name "
                   << this -> getName()
                   << " index is " << i
                   << " and array size is " << this -> getSize() );
   // TODO: add static assert - this does not make sense for tnlCudaDevice
   return Device :: getArrayElementReference( this -> data, i );
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
   Device :: template memcpy< Element,
                              Index,
                              Device >
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
   Device :: template memcpy< typename Array :: ElementType,
                              typename Array :: IndexType,
                              typename Array :: DeviceType >
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
   return Device :: template memcmp< typename Array :: ElementType,
                                     typename Array :: IndexType,
                                     typename Array :: DeviceType >
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
   Device :: template memset< Element, Index >
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
   if( ! file. write< const Index, Device >( &this -> size ) )
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

//}; // namespace implementation


#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

extern template class tnlSharedArray< float, tnlHost, int >;
extern template class tnlSharedArray< double, tnlHost, int >;
extern template class tnlSharedArray< float, tnlHost, long int >;
extern template class tnlSharedArray< double, tnlHost, long int >;

#ifdef HAVE_CUDA
extern template class tnlSharedArray< float, tnlCuda, int >;
extern template class tnlSharedArray< double, tnlCuda, int >;
extern template class tnlSharedArray< float, tnlCuda, long int >;
extern template class tnlSharedArray< double, tnlCuda, long int >;
#endif

#endif

#endif /* TNLSHAREDARRAY_H_IMPLEMENTATION */
