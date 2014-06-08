/***************************************************************************
                          tnlArray_impl.h  -  description
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

#include <iostream>
#include <core/tnlAssert.h>
#include <core/tnlFile.h>
#include <core/mfuncs.h>
#include <core/param-types.h>
#include <core/arrays/tnlArrayOperations.h>

using namespace std;

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
tnlString tnlArray< Element, Device, Index > :: getType()
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
tnlString tnlArray< Element, Device, Index > :: getTypeVirtual() const
{
   return this->getType();
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
   if( this->size == size ) return true;
   if( this->data )
   {
      tnlArrayOperations< Device >::freeMemory( this->data );
      this->data = 0;
   }
   this->size = size;
   tnlArrayOperations< Device >::allocateMemory( this->data, size );
   if( ! this->data )
   {
      cerr << "I am not able to allocate new array with size "
           << ( double ) this->size * sizeof( ElementType ) / 1.0e9 << " GB for "
           << this->getName() << "." << endl;
      this -> size = 0;
      return false;
   }
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
                   << "Array name:" << array. getName()
                   << "Array size: " << array. getSize() << endl );
   return setSize( array.getSize() );
};

template< typename Element,
          typename Device,
          typename Index >
void tnlArray< Element, Device, Index > :: swap( tnlArray< Element, Device, Index >& array )
{
   ::swap( this->size, array.size );
   ::swap( this->data, array.data );
};

template< typename Element,
          typename Device,
          typename Index >
void tnlArray< Element, Device, Index > :: reset()
{
   this->size = 0;
   tnlArrayOperations< Device >::freeMemory( this->data );
   this->data = 0;
};

template< typename Element,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Index tnlArray< Element, Device, Index > :: getSize() const
{
   return this -> size;
}

template< typename Element,
           typename Device,
           typename Index >
void tnlArray< Element, Device, Index > :: setElement( const Index i, const Element& x )
{
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for setElement method in tnlArray with name "
                   << this -> getName()
                   << " index is " << i
                   << " and array size is " << this -> getSize() );
   return tnlArrayOperations< Device > :: setMemoryElement( &( this -> data[ i ] ), x );
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
   return tnlArrayOperations< Device > :: getMemoryElement( & ( this -> data[ i ] ) );
};

template< typename Element,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Element& tnlArray< Element, Device, Index > :: operator[] ( Index i )
{
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for operator[] in tnlArray with name "
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
const Element& tnlArray< Element, Device, Index > :: operator[] ( Index i ) const
{
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for operator[] in tnlArray with name "
                   << this -> getName()
                   << " index is " << i
                   << " and array size is " << this -> getSize() );
   return this->data[ i ];
};

template< typename Element,
           typename Device,
           typename Index >
tnlArray< Element, Device, Index >&
   tnlArray< Element, Device, Index > :: operator = ( const tnlArray< Element, Device, Index >& array )
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
tnlArray< Element, Device, Index >&
   tnlArray< Element, Device, Index > :: operator = ( const Array& array )
{
   tnlAssert( array. getSize() == this -> getSize(),
           cerr << "Source name: " << array. getName() << endl
                << "Source size: " << array. getSize() << endl
                << "Target name: " << this -> getName() << endl
                << "Target size: " << this -> getSize() << endl );
   tnlArrayOperations< Device,
                       typename Array :: DeviceType > ::
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
bool tnlArray< Element, Device, Index > :: operator == ( const Array& array ) const
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
bool tnlArray< Element, Device, Index > :: operator != ( const Array& array ) const
{
   return ! ( ( *this ) == array );
}


template< typename Element,
          typename Device,
          typename Index >
void tnlArray< Element, Device, Index > :: setValue( const Element& e )
{
   tnlAssert( this -> size != 0,
              cerr << "Array name is " << this -> getName() );
   tnlArrayOperations< Device > :: setMemory( this -> getData(), e, this -> getSize() );
}

template< typename Element,
           typename Device,
           typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
const Element* tnlArray< Element, Device, Index > :: getData() const
{
   return this -> data;
}

template< typename Element,
           typename Device,
           typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
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
#ifdef HAVE_NOT_CXX11
   if( ! file. write< const Index, tnlHost >( &this -> size ) )
      return false;
#else            
   if( ! file. write( &this -> size ) )
      return false;
#endif      
   if( this -> size != 0 && ! file. write< Element, Device, Index >( this -> data, this -> size ) )
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
   Index _size;
#ifdef HAVE_NOT_CXX11
   if( ! file. read< Index, tnlHost >( &_size ) )
      return false;
#else   
   if( ! file. read( &_size ) )
      return false;
#endif      
   if( _size < 0 )
   {
      cerr << "Error: The size " << _size << " of the file is not a positive number or zero." << endl;
      return false;
   }
   if( _size )
   {
      setSize( _size );
      if( ! file. read< Element, Device, Index >( this -> data, this -> size ) )
      {
         cerr << "I was not able to READ tnlArray " << this -> getName()
              << " with size " << this -> getSize() << endl;
         return false;
      }
   }
   return true;
}

template< typename Element,
          typename Device,
          typename Index >
bool tnlArray< Element, Device, Index > :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
}

template< typename Element,
          typename Device,
          typename Index >
bool tnlArray< Element, Device, Index > :: load( const tnlString& fileName )
{
   return tnlObject :: load( fileName );
}


template< typename Element,
          typename Device,
          typename Index >
tnlArray< Element, Device, Index > :: ~tnlArray()
{
   if( this -> data )
      tnlArrayOperations< Device > :: freeMemory( this -> data );
}

template< typename Element, typename Device, typename Index >
ostream& operator << ( ostream& str, const tnlArray< Element, Device, Index >& v )
{
   str << "[ ";
   if( v.getSize() > 0 )
   {
      str << v.getElement( 0 );
      for( Index i = 1; i < v.getSize(); i++ )
         str << ", " << v. getElement( i );
   }
   str << " ]";
}


#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

// TODO: this does not work with CUDA 5.5 - fix it later

/*extern template class tnlArray< float, tnlHost, int >;
extern template class tnlArray< double, tnlHost, int >;
extern template class tnlArray< float, tnlHost, long int >;
extern template class tnlArray< double, tnlHost, long int >;*/

#ifdef HAVE_CUDA
/*extern template class tnlArray< float, tnlCuda, int >;
extern template class tnlArray< double, tnlCuda, int >;
extern template class tnlArray< float, tnlCuda, long int >;
extern template class tnlArray< double, tnlCuda, long int >;*/
#endif

#endif

#endif /* TNLARRAY_H_IMPLEMENTATION */
