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
#include <implementation/core/arrays/tnlArrayIO.h>

using namespace std;

#ifdef HAVE_CUDA
__device__ 
inline double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = ( unsigned long long int* ) address;
    unsigned long long int old = *address_as_ull, assumed;
    do 
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double( assumed ) ) );
    } 
    while( assumed != old );
    return __longlong_as_double( old );
}
#endif

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
                     ::getType< Element >() + ", " +
                     Device :: getDeviceType() + ", " +
                     ::getType< Index >() +
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
#ifdef HAVE_CUDA
#else
   tnlAssert( size >= 0,
              cerr << "You try to set size of tnlArray to negative value."
                   << "Name: " << this -> getName() << endl
                   << "New size: " << size << endl );
#endif

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
#ifdef HAVE_CUDA
#else
   tnlAssert( array. getSize() >= 0,
              cerr << "You try to set size of tnlArray to negative value."
                   << "Name: " << this -> getName() << endl
                   << "Array name:" << array. getName()
                   << "Array size: " << array. getSize() << endl );
#endif

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
#ifdef HAVE_CUDA
#else
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for setElement method in tnlArray with name "
                   << this -> getName()
                   << " index is " << i
                   << " and array size is " << this -> getSize() );
#endif

   return tnlArrayOperations< Device > :: setMemoryElement( &( this -> data[ i ] ), x );
};

template< typename Element,
           typename Device,
           typename Index >
Element tnlArray< Element, Device, Index > :: getElement( Index i ) const
{
#ifdef HAVE_CUDA
#else
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for getElement method in tnlArray with name "
                   << this -> getName()
                   << " index is " << i
                   << " and array size is " << this -> getSize() );
#endif

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
#ifdef HAV_CUDA
#else
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for operator[] in tnlArray with name "
                   << this -> getName()
                   << " index is " << i
                   << " and array size is " << this -> getSize() );
#endif

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
#ifdef HAVE_CUDA
#else
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for operator[] in tnlArray with name "
                   << this -> getName()
                   << " index is " << i
                   << " and array size is " << this -> getSize() );
#endif

   return this->data[ i ];
};

template< typename Element,
           typename Device,
           typename Index >
tnlArray< Element, Device, Index >&
   tnlArray< Element, Device, Index > :: operator = ( const tnlArray< Element, Device, Index >& array )
{
#ifdef HAVE_CUDA
#else
   tnlAssert( array. getSize() == this -> getSize(),
           cerr << "Source name: " << array. getName() << endl
                << "Source size: " << array. getSize() << endl
                << "Target name: " << this -> getName() << endl
                << "Target size: " << this -> getSize() << endl );
#endif

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
#ifdef HAVE_CUDA
#else
   tnlAssert( array. getSize() == this -> getSize(),
           cerr << "Source name: " << array. getName() << endl
                << "Source size: " << array. getSize() << endl
                << "Target name: " << this -> getName() << endl
                << "Target size: " << this -> getSize() << endl );
#endif

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
   if( ! tnlObject :: save( file ) )
      return false;
#ifdef HAVE_NOT_CXX11
   if( ! file. write< const Index, tnlHost >( &this -> size ) )
      return false;
#else            
   if( ! file. write( &this -> size ) )
      return false;
#endif      
   if( this -> size != 0 && ! tnlArrayIO< Element, Device, Index >::save( file, this -> data, this -> size ) )
   {
      cerr << "I was not able to save " << this->getType()
           << " " << this -> getName()
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
   setSize( _size );
   if( _size )
   {
      if( ! tnlArrayIO< Element, Device, Index >::load( file, this -> data, this -> size ) )
      {
         cerr << "I was not able to load " << this->getType()
                    << " " << this -> getName()
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
   return str;
}

#ifdef HAVE_CUDA
template< typename Element, typename Device, typename Index >
__device__
void tnlArray< Element, Device, Index >::add( const IndexType pos, const ElementType& val ) const
{
    atomicAdd( &this->data[ pos ], val );
}
#endif


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
