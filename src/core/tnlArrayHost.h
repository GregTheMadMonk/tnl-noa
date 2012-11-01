/***************************************************************************
                          tnlArrayHost.h -  description
                             -------------------
    begin                : Jul 4, 2012
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

#ifndef TNLARRAYMANAGERHOST_H_
#define TNLARRAYMANAGERHOST_H_

#include <cstring>
#include <core/tnlArrayBase.h>
#include <debug/tnlDebug.h>

using namespace std;

template< typename ElementType, typename Device, typename IndexType > class tnlArray;

template< typename ElementType, typename IndexType > class tnlArray< ElementType, tnlCuda, IndexType >;

template< typename ElementType, typename IndexType > class tnlArray< ElementType, tnlHost, IndexType > : public tnlArrayBase< ElementType, IndexType >
{
   //! We do not allow constructor without parameters.
   tnlArray();

   /*!
    * We do not allow copy constructors as well to avoid having two
    * arrays with the same name.
    */

   public:

   //! Basic constructor with given size
   tnlArray( const tnlString& name, IndexType _size = 0 );

   //! Constructor with another array as template
   tnlArray( const tnlString& name, const tnlArray< ElementType, tnlHost, IndexType >& v );

   //! Constructor with another array as template
   tnlArray( const tnlString& name, const tnlArray< ElementType, tnlCuda, IndexType >& v );
   
   tnlArray( const tnlArray< ElementType, tnlHost, IndexType >& v );

   tnlArray( const tnlArray< ElementType, tnlCuda, IndexType >& v );

   //! Use this if you want to insert some data in this array.
   /*! The data will not be deallocated by the destructor.
    *  Once setSize method is called the array forgets the shared data.
    */
   virtual void setSharedData( ElementType* _data, const IndexType _size );

   //! Set size of the array and allocate necessary amount of the memory.
   bool setSize( IndexType _size );

   //! Set size of the array using another array as a template
   bool setLike( const tnlArray< ElementType, tnlHost, IndexType >& v );

   //! Set size of the array using another array as a template
   bool setLike( const tnlArray< ElementType, tnlCuda, IndexType >& v );

   /*!
    * Free allocated memory
    */
   void reset();

   /*!
    * Swaps data between two array managers
    */
   void swap( tnlArray< ElementType, tnlHost, IndexType >& u );

   //! Operator = for copying data from another array on host.
   tnlArray< ElementType, tnlHost, IndexType >& operator = ( const tnlArray< ElementType, tnlHost, IndexType >& array );

   //! Operator = for copying data from another array on CUDA device.
   tnlArray< ElementType, tnlHost, IndexType >& operator =  ( const tnlArray< ElementType, tnlCuda, IndexType >& cuda_array );

   //! Operator = with different element and index types.
   /*!
    * It might by useful for example for copying data from array of floats to
    * array of doubles.
    */
   template< typename ElementType2, typename IndexType2 >
   tnlArray< ElementType, tnlHost, IndexType >& operator = ( const tnlArray< ElementType2, tnlHost, IndexType2 >& a );

   //! Returns type of this array written in a form of C++ template type.
   tnlString getType() const;

   void setElement( IndexType i, ElementType d );

   ElementType getElement( IndexType i ) const;

   ElementType& operator[] ( IndexType i );

   const ElementType& operator[] ( IndexType i ) const;

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );

   virtual ~tnlArray();

};

template< typename ElementType, typename IndexType >
ostream& operator << ( ostream& str, const tnlArray< ElementType, tnlHost, IndexType >& a );

template< typename ElementType, typename IndexType >
tnlArray< ElementType, tnlHost, IndexType > :: tnlArray( const tnlString& name, IndexType _size )
: tnlArrayBase< ElementType, IndexType >( name )
{
   setSize( _size );
};

template< typename ElementType, typename IndexType >
tnlArray< ElementType, tnlHost, IndexType > :: tnlArray( const tnlString& name, const tnlArray& a )
: tnlArrayBase< ElementType, IndexType >( name )
{
   setSize( a. getSize() );
   ( * this ) = a;
};

template< typename ElementType, typename IndexType >
tnlArray< ElementType, tnlHost, IndexType > :: tnlArray( const tnlString& name, const tnlArray< ElementType, tnlCuda, IndexType >& a )
: tnlArrayBase< ElementType, IndexType >( name )
{
  setSize( a. getSize() );
   ( * this ) = a;
};
   
template< typename ElementType, typename IndexType >
tnlArray< ElementType, tnlHost, IndexType > :: tnlArray( const tnlArray< ElementType, tnlHost, IndexType >& v )
: tnlArrayBase< ElementType, IndexType >( "copy of " + v. name )
{
   setSize( v. getSize() );
   ( * this ) = v;
}

template< typename ElementType, typename IndexType >
tnlArray< ElementType, tnlHost, IndexType > :: tnlArray( const tnlArray< ElementType, tnlCuda, IndexType >& v )
: tnlArrayBase< ElementType, IndexType >( "copy of " + v. name )
{
   setSize( v. getSize );
   ( * this ) = v;
}

template< typename ElementType, typename IndexType > bool tnlArray< ElementType, tnlHost, IndexType > :: setSize( IndexType _size )
{
   dbgFunctionName( "tnlArray< ElementType, tnlHost, IndexType >", "setSize" );
   tnlAssert( _size >= 0,
            cerr << "You try to set size of tnlArray to negative value."
                 << "Name: " << this -> getName() << endl
                 << "New size: " << _size << endl );
   dbgExpr( this -> getName() );
   /* In the case that we run without active macro tnlAssert
    * we write at least warning.
    */
   if( _size < 0 )
   {
      cerr << "Negative size " << _size << " was passed to tnlArray " << this -> getName() << "." << endl;
      return false;
   }

   if( this -> size && this -> size == _size && ! this -> shared_data ) return true;
   if( this -> data && ! this -> shared_data )
   {
      delete[] -- ( this -> data );
      this -> data = 0;
   }
   this -> size = _size;
   this -> data = new ElementType[ this -> size + 1 ];
   this -> shared_data = false;
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

template< typename ElementType, typename IndexType >
void tnlArray< ElementType, tnlHost, IndexType > :: setSharedData( ElementType* _data, const IndexType _size )
{
   /****
    * First let us touch the last element to see what happens.
    * If the size or data are not set properly we may get SIGSEGV.
    * It is better to find it out as soon as possible.
    */
   ElementType a = _data[ _size - 1 ];

   if( this -> data && ! this -> shared_data ) delete[] -- this -> data;
   this -> data = _data;
   this -> shared_data = true;
   this -> size = _size;
};

template< typename ElementType, typename IndexType >
bool tnlArray< ElementType, tnlHost, IndexType > :: setLike( const tnlArray< ElementType, tnlHost, IndexType >& a )
{
   return setSize( a. getSize() );
};

template< typename ElementType, typename IndexType >
bool tnlArray< ElementType, tnlHost, IndexType > :: setLike( const tnlArray< ElementType, tnlCuda, IndexType >& a )
{
   return setSize( a. getSize() );
};

template< typename ElementType, typename IndexType >
void tnlArray< ElementType, tnlHost, IndexType > :: reset()
{
   dbgFunctionName( "tnlArray< ElementType, tnlHost, IndexType >", "reset" );
   dbgExpr( this -> getName() );
   setSize( 0 );
};


template< typename ElementType, typename IndexType >
void tnlArray< ElementType, tnlHost, IndexType > :: swap( tnlArray< ElementType, tnlHost, IndexType >& v )
{
   tnlAssert( this -> getSize() > 0, );
   tnlAssert( this -> getSize() == v. getSize(),
              cerr << "You try to swap two arrays with different sizes." << endl
                   << "The first one is " << this -> getName() << " with size " << this -> getSize()
                   << " while the second one is " << v. getName() << " with size " << v. getSize() << "." );

   ElementType* auxData = this -> data;
   this -> data = v. data;
   v. data = auxData;
   bool auxShared = this -> shared_data;
   this -> shared_data = v. shared_data;
   v. shared_data = auxShared;
};

template< typename ElementType, typename IndexType >
tnlString tnlArray< ElementType, tnlHost, IndexType > :: getType() const
{
   return tnlString( "tnlArray< " ) + getParameterType< ElementType >() + tnlString( ", tnlHost >" );
};

template< typename ElementType, typename IndexType >
tnlArray< ElementType, tnlHost, IndexType >& tnlArray< ElementType, tnlHost, IndexType > :: operator = ( const tnlArray< ElementType, tnlHost, IndexType >& a )
{
   tnlAssert( a. getSize() == this -> getSize(),
           cerr << "Source name: " << a. getName() << endl
                << "Source size: " << a. getSize() << endl
                << "Target name: " << this -> getName() << endl
                << "Target size: " << this -> getSize() << endl );

   /*memcpy( this -> data,
           a. getData(),
           this -> getSize() * sizeof( ElementType ) );*/
   for( IndexType i = 0; i < this -> getSize(); i ++ )
      ( *this )[ i ] = a[ i ];
   return ( *this );
};

template< typename ElementType, typename IndexType >
   template< typename ElementType2, typename IndexType2 >
tnlArray< ElementType, tnlHost, IndexType >& tnlArray< ElementType, tnlHost, IndexType > :: operator = ( const tnlArray< ElementType2, tnlHost, IndexType2 >& a )
{
   tnlAssert( a. getSize() == this -> getSize(),
           cerr << "Source name: " << a. getName() << endl
                << "Source size: " << a. getSize() << endl
                << "Target name: " << this -> getName() << endl
                << "Target size: " << this -> getSize() << endl );

   for( IndexType i = 0; i < this -> getSize(); i ++ )
      this -> data[ i ] = ( ElementType ) a. getElement( ( IndexType2 ) i );
   return ( *this );
};

template< typename ElementType, typename IndexType >
tnlArray< ElementType, tnlHost, IndexType >& tnlArray< ElementType, tnlHost, IndexType > :: operator =  ( const tnlArray< ElementType, tnlCuda, IndexType >& cuda_array )
{
#ifdef HAVE_CUDA
   tnlAssert( cuda_array. getSize() == this -> getSize(),
              cerr << "You try to copy one array to another with different size." << endl
                   << "The CUDA source array " << cuda_vector. getName() << " size is: " << cuda_vector. getSize() << endl                 << " this -> getSize() = " << this -> getSize()
                   << "The target array " << this -> getName() << " size is " << this -> getSize() << endl; );
   if( cudaMemcpy( this -> data,
                   cuda_array. getData(),
                   this -> getSize() * sizeof( ElementType ),
                   cudaMemcpyDeviceToHost ) != cudaSuccess )
   {
      cerr << "Transfer of data from CUDA device ( " << cuda_array. getName()
           << " ) to CUDA host ( " << this -> getName() << " ) failed." << endl;
      return *this;
   }
   if( cuda_array. getSafeMode() )
      cudaThreadSynchronize();
   return *this;
#else
   cerr << "CUDA support is missing in this system." << endl;
   return *this;
#endif
};

template< typename ElementType, typename IndexType >
void tnlArray< ElementType, tnlHost, IndexType > :: setElement( IndexType i, ElementType d )
{
   tnlAssert( this -> size != 0,
              cerr << "Vector name is " << this -> getName() );
   tnlAssert( i < this -> size,
            cerr << "You try to set non-existing element of the following vector."
                 << "Name: " << this -> getName() << endl
                 << "Size: " << this -> size << endl
                 << "Element number: " << i << endl; );
   this -> data[ i ] = d;
};

template< typename ElementType, typename IndexType >
ElementType tnlArray< ElementType, tnlHost, IndexType > :: getElement( IndexType i ) const
{
   tnlAssert( this -> size != 0,
              cerr << "Vector name is " << this -> getName() );
   tnlAssert( i < this -> size,
            cerr << "You try to get non-existing element of the following vector."
                 << "Name: " << this -> getName() << endl
                 << "Size: " << this -> size << endl
                 << "Element number: " << i << endl; );
   return this -> data[ i ];
};

template< typename ElementType, typename IndexType >
ElementType& tnlArray< ElementType, tnlHost, IndexType > :: operator[] ( IndexType i )
{
    tnlAssert( i < this -> size,
           cerr << "Name: " << this -> getName() << endl
                << "Size: " << this -> size << endl
                << "i = " << i << endl; );
   return this -> data[ i ];
};

template< typename ElementType, typename IndexType >
const ElementType& tnlArray< ElementType, tnlHost, IndexType > :: operator[] ( IndexType i ) const
{
   tnlAssert( i < this -> size,
           cerr << "Name: " << this -> getName() << endl
                << "Size: " << this -> size << endl
                << "i = " << i << endl; );
   return this -> data[ i ];
};

template< typename ElementType, typename IndexType >
bool tnlArray< ElementType, tnlHost, IndexType > :: save( tnlFile& file ) const
{
   tnlAssert( this -> size != 0,
              cerr << "You try to save empty vector. Its name is " << this -> getName() );
   if( ! tnlObject :: save( file ) )
      return false;
   if( ! file. write( &this -> size, 1 ) )
      return false;
   if( ! file. write( this -> data, this -> size ) )
   {
      cerr << "I was not able to WRITE the long vector " << this -> getName()
           << " with size " << this -> getSize() << endl;
      return false;
   }
   return true;
};

template< typename ElementType, typename IndexType >
bool tnlArray< ElementType, tnlHost, IndexType > :: load( tnlFile& file )
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
   if( ! file. read( this -> data, this -> size ) )
   {
      cerr << "I was not able to READ the long vector " << this -> getName()
           << " with size " << this -> getSize() << endl;
      return false;
   }
   return true;
};

template< typename ElementType, typename IndexType >
tnlArray< ElementType, tnlHost, IndexType > :: ~tnlArray()
{
   if( this -> data && ! this -> shared_data ) delete[] -- this -> data;
};

template< typename ElementType, typename IndexType >
ostream& operator << ( ostream& str, const tnlArray< ElementType, tnlHost, IndexType >& vec )
{
   for( IndexType i = 0; i < vec. getSize(); i ++ )
      str << vec[ i ] << " ";
   return str;
};

#endif /* TNLARRAYMANAGERHOST_H_ */
