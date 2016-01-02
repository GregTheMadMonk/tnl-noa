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
#include <core/arrays/tnlArrayIO.h>

#include "tnlArray.h"

using namespace std;

template< typename Element,
           typename Device,
           typename Index >
tnlArray< Element, Device, Index >::
tnlArray()
: size( 0 ),
  data( 0 ),
  allocationPointer( 0 ),
  referenceCounter( 0 )
{
};

template< typename Element,
           typename Device,
           typename Index >
tnlArray< Element, Device, Index >::
tnlArray( const IndexType& size )
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
tnlArray< Element, Device, Index >::
tnlArray( Element* data,
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
tnlArray< Element, Device, Index >::
tnlArray( tnlArray< Element, Device, Index >& array,
          const IndexType& begin,
          const IndexType& size )
: size( size ),
  data( &array.getData()[ begin ] ),
  allocationPointer( array.allocationPointer ),
  referenceCounter( 0 )
{
   tnlAssert( begin < array.getSize(),
              std::cerr << " begin = " << begin << " array.getSize() = " << array.getSize() );
   tnlAssert( begin + size  < array.getSize(),
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
tnlString 
tnlArray< Element, Device, Index >::
getType()
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
tnlString 
tnlArray< Element, Device, Index >::
getTypeVirtual() const
{
   return this->getType();
};

template< typename Element,
           typename Device,
           typename Index >
tnlString
tnlArray< Element, Device, Index >::
getSerializationType()
{
   return HostType::getType();
};

template< typename Element,
           typename Device,
           typename Index >
tnlString 
tnlArray< Element, Device, Index >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Element,
           typename Device,
           typename Index >
void
tnlArray< Element, Device, Index >::
releaseData() const
{
   if( this->referenceCounter )
   {
      if( --*this->referenceCounter == 0 )
      {
         tnlArrayOperations< Device >::freeMemory( this->allocationPointer );
         delete this->referenceCounter;
      }
   }
   else
      if( allocationPointer )
         tnlArrayOperations< Device >::freeMemory( this->allocationPointer );
   this->allocationPointer = 0;
   this->data = 0;
   this->size = 0;
   this->referenceCounter = 0;
}

template< typename Element,
          typename Device,
          typename Index >
bool
tnlArray< Element, Device, Index >::
setSize( const Index size )
{
   tnlAssert( size >= 0,
              cerr << "You try to set size of tnlArray to negative value."
                   << "New size: " << size << endl );
   if( this->size == size && allocationPointer && ! referenceCounter ) return true;
   this->releaseData();
   tnlArrayOperations< Device >::allocateMemory( this->allocationPointer, size );
   this->data = this->allocationPointer;
   this->size = size;
   if( ! this->allocationPointer )
   {
      cerr << "I am not able to allocate new array with size "
           << ( double ) this->size * sizeof( ElementType ) / 1.0e9 << " GB." << endl;
      this -> size = 0;
      return false;
   }   
   return true;
};

template< typename Element,
           typename Device,
           typename Index >
   template< typename Array >
bool
tnlArray< Element, Device, Index >::
setLike( const Array& array )
{
   tnlAssert( array. getSize() >= 0,
              cerr << "You try to set size of tnlArray to negative value."
                   << "Array size: " << array. getSize() << endl );
   return setSize( array.getSize() );
};

template< typename Element,
           typename Device,
           typename Index >
void 
tnlArray< Element, Device, Index >::
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
void
tnlArray< Element, Device, Index >::
bind( const tnlArray< Element, Device, Index >& array,
      const IndexType& begin,
      const IndexType& size )
{
   tnlAssert( begin <= array.getSize(),
              std::cerr << " begin = " << begin << " array.getSize() = " << array.getSize() );
   tnlAssert( begin + size  <= array.getSize(),
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
   template< int Size >
void
tnlArray< Element, Device, Index >::
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
tnlArray< Element, Device, Index >::
swap( tnlArray< Element, Device, Index >& array )
{
   ::swap( this->size, array.size );
   ::swap( this->data, array.data );
   ::swap( this->allocationPointer, array.allocationPointer );
   ::swap( this->referenceCounter, array.referenceCounter );
};

template< typename Element,
          typename Device,
          typename Index >
void 
tnlArray< Element, Device, Index >::
reset()
{
   this->releaseData();
};

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
Index 
tnlArray< Element, Device, Index >::
getSize() const
{
   return this -> size;
}

template< typename Element,
           typename Device,
           typename Index >
void
tnlArray< Element, Device, Index >::
setElement( const Index& i, const Element& x )
{
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for setElement method in tnlArray "
                   << " index is " << i
                   << " and array size is " << this -> getSize() );
   return tnlArrayOperations< Device > :: setMemoryElement( &( this -> data[ i ] ), x );
};

template< typename Element,
           typename Device,
           typename Index >
Element
tnlArray< Element, Device, Index >::
getElement( const Index& i ) const
{
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for getElement method in tnlArray "
                   << " index is " << i
                   << " and array size is " << this -> getSize() );
   return tnlArrayOperations< Device > :: getMemoryElement( & ( this -> data[ i ] ) );
};

template< typename Element,
          typename Device,
          typename Index >
__cuda_callable__
inline Element&
tnlArray< Element, Device, Index >::
operator[] ( const Index& i )
{
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for operator[] in tnlArray "
                   << " index is " << i
                   << " and array size is " << this -> getSize() );
   return this->data[ i ];
};

template< typename Element,
           typename Device,
           typename Index >
__cuda_callable__
inline const Element& 
tnlArray< Element, Device, Index >::
operator[] ( const Index& i ) const
{
   tnlAssert( 0 <= i && i < this -> getSize(),
              cerr << "Wrong index for operator[] in tnlArray "
                   << " index is " << i
                   << " and array size is " << this -> getSize() );
   return this->data[ i ];
};

template< typename Element,
           typename Device,
           typename Index >
tnlArray< Element, Device, Index >&
tnlArray< Element, Device, Index >::
operator = ( const tnlArray< Element, Device, Index >& array )
{
   tnlAssert( array. getSize() == this -> getSize(),
           cerr << "Source size: " << array. getSize() << endl
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
tnlArray< Element, Device, Index >::
operator = ( const Array& array )
{
   tnlAssert( array. getSize() == this -> getSize(),
           cerr << "Source size: " << array. getSize() << endl
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
bool
tnlArray< Element, Device, Index >::
operator == ( const Array& array ) const
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
   tnlAssert( this->getData(),);
   tnlArrayOperations< Device > :: setMemory( this -> getData(), e, this -> getSize() );
}

template< typename Element,
           typename Device,
           typename Index >
__cuda_callable__
const Element* tnlArray< Element, Device, Index > :: getData() const
{
   return this -> data;
}

template< typename Element,
           typename Device,
           typename Index >
__cuda_callable__
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
   if( ! file. write< const Index, tnlHost >( &this->size ) )
      return false;
#else            
   if( ! file. write( &this->size ) )
      return false;
#endif
   if( this -> size != 0 && ! tnlArrayIO< Element, Device, Index >::save( file, this -> data, this -> size ) )
   {
      cerr << "I was not able to save " << this->getType()
           << " with size " << this -> getSize() << endl;
      return false;
   }
   return true;
};

template< typename Element,
          typename Device,
          typename Index >
bool
tnlArray< Element, Device, Index >::
load( tnlFile& file )
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
                    << " with size " << this -> getSize() << endl;
         return false;
      }
   }
   return true;
}

template< typename Element,
          typename Device,
          typename Index >
bool
tnlArray< Element, Device, Index >::
boundLoad( tnlFile& file )
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
      if( ! tnlArrayIO< Element, Device, Index >::load( file, this -> data, this -> size ) )
      {
         cerr << "I was not able to load " << this->getType()
                    << " with size " << this -> getSize() << endl;
         return false;
      }
   }
   return true;
}

template< typename Element,
          typename Device,
          typename Index >
bool
tnlArray< Element, Device, Index >::
boundLoad( const tnlString& fileName )
{
   tnlFile file;
   if( ! file. open( fileName, tnlReadMode ) )
   {
      cerr << "I am not bale to open the file " << fileName << " for reading." << endl;
      return false;
   }
   if( ! this->boundLoad( file ) )
      return false;
   if( ! file. close() )
   {
      cerr << "An error occurred when I was closing the file " << fileName << "." << endl;
      return false;
   }
   return true;   
}

template< typename Element,
          typename Device,
          typename Index >
tnlArray< Element, Device, Index >::
~tnlArray()
{
   this->releaseData();
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


#ifdef UNDEF //TEMPLATE_EXPLICIT_INSTANTIATION

// TODO: this does not work with CUDA 5.5 - fix it later

#ifdef INSTANTIATE_FLOAT
extern template class tnlArray< float, tnlHost, int >;
#endif
extern template class tnlArray< double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlArray< long double, tnlHost, int >;
#endif

#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class tnlArray< float, tnlHost, long int >;
#endif
extern template class tnlArray< double, tnlHost, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlArray< long double, tnlHost, long int >;
#endif
#endif

#ifdef HAVE_CUDA
/*
 #ifdef INSTANTIATE_FLOAT
 extern template class tnlArray< float, tnlCuda, int >;
 #endif
 extern template class tnlArray< double, tnlCuda, int >;
 #ifdef INSTANTIATE_FLOAT
 extern template class tnlArray< float, tnlCuda, long int >;
 #endif
 extern template class tnlArray< double, tnlCuda, long int >;*/
#endif

#endif

#endif /* TNLARRAY_H_IMPLEMENTATION */
