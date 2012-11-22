/***************************************************************************
                          tnlArray.h  -  description
                             -------------------
    begin                : Nov 8, 2012
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

#ifndef TNLARRAY_H_IMPLEMENTATION
#define TNLARRAY_H_IMPLEMENTATION

#include <core/tnlFile.h>
#include <core/mfuncs.h>
#include <core/param-types.h>

//namespace implementation
//{

template< typename Element,
          typename Device,
          typename Index >
tnlArray< Element, Device, Index > :: tnlArray()
: size( 0 ), data( 0 )
{
};

template< typename Element,
          typename Device,
          typename Index >
tnlArray< Element, Device, Index > :: tnlArray( const tnlString& name )
: size( 0 ), data( 0 )
{
   this -> setName( name );
};

template< typename Element,
          typename Device,
          typename Index >
tnlString tnlArray< Element, Device, Index > :: getType() const
{
   return tnlString( "tnlArray< " ) +
                     getParameterType< Element >() +
                     Device :: getDeviceType() +
                     getParameterType< Index >() +
                     " >";
};

template< typename Element,
          typename Device,
          typename Index >
bool tnlArray< Element, Device, Index > :: setSize( const Index size )
{
   tnlAssert( size >= 0,
              cerr << "You try to set size of tnlArray to negative value."
                   << "Name: " << this -> getName() << endl
                   << "New size: " << size << endl );
   if( this -> size && this -> size == size ) return true;
   if( this -> data )
   {
      Device :: freeMemory( this -> data );
      this -> data = 0;
   }
   this -> size = size;
   Device :: allocateMemory( this -> data, size + 1 );
   if( ! this -> data )
   {
      cerr << "I am not able to allocate new array with size "
           << ( double ) this -> size * sizeof( ElementType ) / 1.0e9 << " GB on host for "
           << this -> getName() << "." << endl;
      this -> size = 0;
      return false;
   }
   ( this -> data ) ++;
   return true;
};

template< typename Element,
          typename Device,
          typename Index >
   template< typename Array >
bool tnlArray< Element, Device, Index > :: setLike( const Array& array )
{
   tnlAssert( array. getSize() >= 0,
              cerr << "You try to set size of tnlArray to negative value."
                   << "Name: " << this -> getName() << endl
                   << "Array name:" << array. getName() <<
                   << "Array size: " << array. getSize() << endl );
   return setSize( array. getSize() );
};

template< typename Element,
          typename Device,
          typename Index >
void tnlArray< Element, Device, Index > :: swap( tnlArray< Element, Device, Index >& array )
{
   swap( this -> size, array. size );
   swap( this -> data, array. data );
};

template< typename Element,
          typename Device,
          typename Index >
void tnlArray< Element, Device, Index > :: reset()
{
   this -> size = 0;
   this -> data = 0;
};

template< typename Element,
          typename Device,
          typename Index >
Index tnlArray< Element, Device, Index > :: getSize() const
{
   return this -> size;
}

template< typename Element,
          typename Device,
          typename Index >
void tnlArray< Element, Device, Index > :: setElement( const Element& x, Index i )
{
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for setElement method in tnlArray with name "
                   << this -> getName()
                   << " index is " << i
                   << " and array size is " << this -> getSize() );
   return Device :: setArrayElement( this -> data, x, i );
};

template< typename Element,
          typename Device,
          typename Index >
Element tnlArray< Element, Device, Index > :: getElement( Index i ) const
{
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for getElement method in tnlArray with name "
                   << this -> getName()
                   << " index is " << i
                   << " and array size is " << this -> getSize() );
   return Device :: getArrayElement( this -> data, i );
};

template< typename Element,
          typename Device,
          typename Index >
Element& tnlArray< Element, Device, Index > :: operator[] ( Index i )
{
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for operator[] in tnlArray with name "
                   << this -> getName()
                   << " index is " << i
                   << " and array size is " << this -> getSize() );
   // TODO: add static assert - this does not make sense for tnlCudaDevice
   return Device :: getArrayElementReference( this -> data, i );
};

template< typename Element,
          typename Device,
          typename Index >
const Element& tnlArray< Element, Device, Index > :: operator[] ( Index i ) const
{
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for operator[] in tnlArray with name "
                   << this -> getName()
                   << " index is " << i
                   << " and array size is " << this -> getSize() );
   // TODO: add static assert - this does not make sense for tnlCudaDevice
   return Device :: getArrayElementReference( this -> data, i );
};

template< typename Element,
          typename Device,
          typename Index >
   template< typename Array >
tnlArray< Element, Device, Index >& tnlArray< Element, Device, Index > :: operator = ( const Array& array )
{
   typedef typename Array :: ElementType ArrayElement;
   typedef typename Array :: IndexType ArrayIndex;
   typedef typename Array :: DeviceType ArrayDevice;
   
   // TODO: check this
   /*STATIC_ASSERT( ElementType == Array :: ElementType &&
                  IndexType == Array :: IndexType,
                  "Cannot assign arrays with different element types or index types (in tnlArray)" );*/
   tnlAssert( array. getSize() == this -> getSize(),
           cerr << "Source name: " << a. getName() << endl
                << "Source size: " << a. getSize() << endl
                << "Target name: " << this -> getName() << endl
                << "Target size: " << this -> getSize() << endl );
   Device :: template memcpy< ArrayElement,
                              ArrayIndex,
                              ArrayDevice >
                   ( this -> getData(),
                     array. getData(),
                     array. getSize() );
   return ( *this );
};

template< typename Element,
          typename Device,
          typename Index >
   template< typename Array >
bool tnlArray< Element, Device, Index > :: operator == ( const Array& array ) const
{
   typedef typename Array :: ElementType ArrayElement;
   typedef typename Array :: IndexType ArrayIndex;
   typedef typename Array :: DeviceType ArrayDevice;
   // TODO: check this
   /*STATIC_ASSERT( ElementType == Array :: ElementType &&
                  IndexType == Array :: IndexType,
                  "Cannot assign arrays with different element types or index types (in tnlArray)" );*/
   tnlAssert( array. getSize() == this -> getSize(),
           cerr << "Source name: " << a. getName() << endl
                << "Source size: " << a. getSize() << endl
                << "Target name: " << this -> getName() << endl
                << "Target size: " << this -> getSize() << endl );
   return Device :: template memcmp< ArrayElement,
                                     ArrayIndex,
                                     ArrayDevice >
                                   ( this -> getData(),
                                     array. getData(),
                                     array. getSize() );
}

template< typename Element,
          typename Device,
          typename Index >
   template< typename Array >
bool tnlArray< Element, Device, Index > :: operator != ( const Array& array ) const
{
   // TODO: check this
   /*STATIC_ASSERT( ElementType == Array :: ElementType &&
                  IndexType == Array :: IndexType,
                  "Cannot assign arrays with different element types or index types (in tnlArray)" );*/
   tnlAssert( array. getSize() == this -> getSize(),
           cerr << "Source name: " << a. getName() << endl
                << "Source size: " << a. getSize() << endl
                << "Target name: " << this -> getName() << endl
                << "Target size: " << this -> getSize() << endl );
   return ! ( ( *this ) == array );
}

template< typename Element,
          typename Device,
          typename Index >
void tnlArray< Element, Device, Index > :: setValue( const Element& e )
{
   tnlAssert( this -> size != 0,
              cerr << "Array name is " << this -> getName() );
   Device :: memset( this -> getData(), this -> getSize(), e );
}

template< typename Element,
          typename Device,
          typename Index >
const Element* tnlArray< Element, Device, Index > :: getData() const
{
   return this -> data;
}

template< typename Element,
          typename Device,
          typename Index >
Element* tnlArray< Element, Device, Index > :: getData()
{
   return this -> data;
}

template< typename Element,
          typename Device,
          typename Index >
tnlArray< Element, Device, Index > :: operator bool() const
{
   return data != 0;
};


template< typename Element,
          typename Device,
          typename Index >
   template< typename IndexType2 >
void tnlArray< Element, Device, Index > :: touch( IndexType2 touches ) const
{
   //TODO: implement
};

template< typename Element,
          typename Device,
          typename Index >
bool tnlArray< Element, Device, Index > :: save( tnlFile& file ) const
{
   tnlAssert( this -> size != 0,
              cerr << "You try to save empty vector. Its name is " << this -> getName() );
   if( ! tnlObject :: save( file ) )
      return false;
   if( ! file. write( &this -> size, 1 ) )
      return false;
   if( ! file. write< Element, Device, Index >( this -> data, this -> size ) )
   {
      cerr << "I was not able to WRITE tnlArray " << this -> getName()
           << " with size " << this -> getSize() << endl;
      return false;
   }
   return true;
};

template< typename Element,
          typename Device,
          typename Index >
bool tnlArray< Element, Device, Index > :: load( tnlFile& file )
{
   if( ! tnlObject :: load( file ) )
      return false;
   int _size;
   if( ! file. read( &_size, 1 ) )
      return false;
   if( _size <= 0 )
   {
      cerr << "Error: The size " << _size << " of the file is not a positive number." << endl;
      return false;
   }
   setSize( _size );
   if( ! file. read< Element, Device, Index >( this -> data, this -> size ) )
   {
      cerr << "I was not able to READ tnlArray " << this -> getName()
           << " with size " << this -> getSize() << endl;
      return false;
   }
   return true;
}

template< typename Element,
          typename Device,
          typename Index >
tnlArray< Element, Device, Index > :: ~tnlArray()
{
   if( this -> data )
      Device :: freeMemory( this -> data );
}

//}; // namespace implementation

#endif /* TNLARRAY_H_IMPLEMENTATION */
