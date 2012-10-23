/***************************************************************************
                          tnlArrayManagerCUDA.h -  description
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

#ifndef TNLARRAYMANAGERCUDA_H_
#define TNLARRAYMANAGERCUDA_H_

/* When we need to transfer data between the GPU and the CPU we use
 * 5 MB buffer. This size should ensure good performance -- see.
 * http://wiki.accelereyes.com/wiki/index.php/GPU_Memory_Transfer
 * Similar constant is defined also in tnlFile.
 */
const int tnlGPUvsCPUTransferBufferSize = 5 * 1 << 20;

template< typename ElementType, typename IndexType >
class tnlArrayManager< ElementType, tnlCuda, IndexType > : public tnlArrayManagerBase< ElementType, IndexType >
{
   //! We do not allow constructor without parameters.
   tnlArrayManager(){};

   /****
    * We do not allow copy constructors as well to avoid having two
    * arrays with the same name.
    */
   tnlArrayManager( const tnlArrayManager< ElementType, tnlHost, IndexType >& v ){};

   tnlArrayManager( const tnlArrayManager< ElementType, tnlCuda, IndexType >& v ){};

   public:

   //! Basic constructor with given size
   tnlArrayManager( const tnlString& name, IndexType _size = 0 );

   //! Constructor with another array as template
   tnlArrayManager( const tnlString& name, const tnlArrayManager< ElementType, tnlHost, IndexType >& v );

   //! Constructor with another array as template
   tnlArrayManager( const tnlString& name, const tnlArrayManager< ElementType, tnlCuda, IndexType >& v );

   //! Use this if you want to insert some data in this array.
   /*! The data will not be deallocated by the destructor.
    *  Once setSize method is called the array forgets the shared data.
    */
   virtual void setSharedData( ElementType* _data, const IndexType _size );

   //! Set size of the array and allocate necessary amount of the memory.
   bool setSize( IndexType _size );

   //! Set size of the array using another array as a template
   bool setLike( const tnlArrayManager< ElementType, tnlHost, IndexType >& v );

   //! Set size of the array using another array as a template
   bool setLike( const tnlArrayManager< ElementType, tnlCuda, IndexType >& v );

   /*!**
    * Free allocated memory
    */
   void reset();

   /*!***
    * Swaps data between two array managers
    */
   void swap( tnlArrayManager< ElementType, tnlCuda, IndexType >& u );

   //! Returns type of this array written in a form of C++ template type.
   tnlString getType() const;

   tnlArrayManager< ElementType, tnlCuda, IndexType >& operator = ( const tnlArrayManager< ElementType, tnlCuda, IndexType >& array );

   tnlArrayManager< ElementType, tnlCuda, IndexType >& operator = ( const tnlArrayManager< ElementType, tnlHost, IndexType >& array );

   template< typename ElementType2, typename IndexType2 >
   tnlArrayManager< ElementType, tnlCuda, IndexType >& operator = ( const tnlArrayManager< ElementType2, tnlCuda, IndexType2 >& array );

   void setElement( IndexType i, ElementType d );

   ElementType getElement( IndexType i ) const;

   ElementType& operator[] ( IndexType i );

   const ElementType& operator[] ( IndexType i ) const;

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );

   virtual ~tnlArrayManager();

};

template< typename ElementType, typename IndexType >
bool operator == ( const tnlArrayManager< ElementType, tnlHost, IndexType >& host_vector,
                   const tnlArrayManager< ElementType, tnlCuda, IndexType >& cuda_vector );

template< typename ElementType, typename IndexType >
bool operator != ( const tnlArrayManager< ElementType, tnlHost, IndexType >& host_vector,
                   const tnlArrayManager< ElementType, tnlCuda, IndexType >& cuda_vector );

template< typename ElementType, typename IndexType >
ostream& operator << ( ostream& str, const tnlArrayManager< ElementType, tnlCuda, IndexType >& vec );

template< typename ElementType, typename IndexType >
tnlArrayManager< ElementType, tnlCuda, IndexType > :: tnlArrayManager( const tnlString& name, IndexType _size )
: tnlArrayManagerBase< ElementType, IndexType >( name )
{
   setSize( _size );
};

template< typename ElementType, typename IndexType >
tnlArrayManager< ElementType, tnlCuda, IndexType > :: tnlArrayManager( const tnlString& name,
                                                        const tnlArrayManager< ElementType, tnlHost, IndexType >& v )
   : tnlArrayManagerBase< ElementType, IndexType >( name )
{
   setSize( v. getSize() );
};

template< typename ElementType, typename IndexType >
tnlArrayManager< ElementType, tnlCuda, IndexType > :: tnlArrayManager( const tnlString& name,
                                                        const tnlArrayManager< ElementType, tnlCuda, IndexType >& v )
   : tnlArrayManagerBase< ElementType, IndexType >( name )
{
   setSize( v. getSize() );
};

template< typename ElementType, typename IndexType >
void tnlArrayManager< ElementType, tnlCuda, IndexType > :: setSharedData( ElementType* _data, const IndexType _size )
{
#ifdef HAVE_CUDA

   if( this -> data &&
     ! this -> shared_data ) cudaFree( this -> data );
   this -> data = _data;
   this -> shared_data = true;
   this -> size = _size;
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
};

template< typename ElementType, typename IndexType >
bool tnlArrayManager< ElementType, tnlCuda, IndexType > :: setSize( IndexType _size )
{
   dbgFunctionName( "tnlArrayManagerCUDA", "setSize" );
   tnlAssert( _size >= 0,
            cerr << "You try to set size of tnlArrayManager to negative value."
                 << "Vector name: " << this -> getName() << endl
                 << "New size: " << _size << endl );
   /* In the case that we run without active macro tnlAssert
    * we will write at least warning.
    */
   if( _size < 0 )
   {
      cerr << "Negative size " << _size << " was passed to tnlArrayManager " << this -> getName() << "." << endl;
      return false;
   }
#ifdef HAVE_CUDA
   dbgCout( "Setting new size to " << _size << " for " << this -> getName() );
   if( this -> size && this -> size == _size && ! this -> shared_data ) return true;
   if( this -> data && ! this -> shared_data )
   {
      dbgCout( "Freeing allocated memory on CUDA device of " << this -> getName() );
      cudaFree( this -> data );
      if( ! checkCUDAError( __FILE__, __LINE__ ) )
      return false;
      this -> data = NULL;
   }
   this -> size = _size;
   this -> shared_data = false;
   if( this -> getSize() != 0 )
   {
      if( cudaMalloc( ( void** ) & this -> data,
                      ( size_t ) this -> size * sizeof( ElementType ) ) != cudaSuccess )
      {
         cerr << "I am not able to allocate new long vector with size " << this -> size * sizeof( ElementType ) << " ("
              << ( double ) this -> size * sizeof( ElementType ) / 1.0e9 << " GB) on CUDA device for "
              << this -> getName() << "." << endl;
         checkCUDAError( __FILE__, __LINE__ );
         this -> data = NULL;
         this -> size = 0;
         return false;
      }
   }
   else
      this -> data = NULL;
   return true;
#else
   //cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
   return false;
#endif
};

template< typename ElementType, typename IndexType >
bool tnlArrayManager< ElementType, tnlCuda, IndexType > :: setLike( const tnlArrayManager< ElementType, tnlHost, IndexType >& v )
{
   return setSize( v. getSize() );
};

template< typename ElementType, typename IndexType >
bool tnlArrayManager< ElementType, tnlCuda, IndexType > :: setLike( const tnlArrayManager< ElementType, tnlCuda, IndexType >& v )
{
   return setSize( v. getSize() );
};

template< typename ElementType, typename IndexType >
void tnlArrayManager< ElementType, tnlCuda, IndexType > :: reset()
{
   dbgFunctionName( "tnlArrayManagerCUDA", "reset" );
#ifdef HAVE_CUDA
   if( this -> data && ! this -> shared_data )
   {
      dbgCout( "Freeing allocated memory on CUDA device of " << this -> getName() );
      cudaFree( this -> data );
      if( ! checkCUDAError( __FILE__, __LINE__ ) )
      this -> data = NULL;
   }
   this -> size = 0;
   this -> shared_data = false;
   this -> data = 0;
#endif
};

template< typename ElementType, typename IndexType >
void tnlArrayManager< ElementType, tnlCuda, IndexType > :: swap( tnlArrayManager< ElementType, tnlCuda, IndexType >& v )
{
   tnlAssert( this -> getSize() > 0, );
   tnlAssert( this -> getSize() == v. getSize(),
              cerr << "You try to swap two long vectors with different sizes." << endl
                   << "The first one is " << this -> getName() << " with size " << this -> getSize()
                   << " while the second one is " << v. getName() << " with size " << v. getSize() << "." );

   std :: swap( this -> data, v. data );
   std :: swap( this -> shared_data, v. shared_data );
};

template< typename ElementType, typename IndexType >
tnlString tnlArrayManager< ElementType, tnlCuda, IndexType > :: getType() const
{
    ElementType t;
    return tnlString( "tnlArrayManager< " ) + tnlString( GetParameterType( t ) ) + tnlString( ", tnlCuda >" );
};

template< typename ElementType, typename IndexType >
tnlArrayManager< ElementType, tnlCuda, IndexType >& tnlArrayManager< ElementType, tnlCuda, IndexType > :: operator = ( const tnlArrayManager< ElementType, tnlHost, IndexType >& rhs_vector )
{
#ifdef HAVE_CUDA
   tnlAssert( rhs_vector. getSize() == this -> getSize(),
           cerr << "Source name: " << rhs_vector. getName() << endl
                << "Source size: " << rhs_vector. getSize() << endl
                << "Target name: " << this -> getName() << endl
                << "Target size: " << this -> getSize() << endl );

   if( cudaMemcpy( this -> data,
                   rhs_vector. getData(),
                   this -> getSize() * sizeof( ElementType ),
                   cudaMemcpyHostToDevice ) != cudaSuccess )
   {
      checkCUDAError( __FILE__, __LINE__ );
      cerr << "Transfer of data from CUDA host ( " << rhs_vector. getName()
           << " ) to CUDA device ( " << this -> getName() << " ) failed." << endl;
      cerr << "Data: " << this -> data << endl;
   }
   if( ! this -> getSafeMode() )
      cudaThreadSynchronize();
   return *this;
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
   return *this;
#endif
};

template< typename ElementType, typename IndexType >
tnlArrayManager< ElementType, tnlCuda, IndexType >& tnlArrayManager< ElementType, tnlCuda, IndexType > :: operator = ( const tnlArrayManager< ElementType, tnlCuda, IndexType >& rhs_vector )
{
#ifdef HAVE_CUDA
    tnlAssert( rhs_vector. getSize() == this -> getSize(),
            cerr << "Source name: " << rhs_vector. getName() << endl
                 << "Source size: " << rhs_vector. getSize() << endl
                 << "Target name: " << this -> getName() << endl
                 << "Target size: " << this -> getSize() << endl );

   if( cudaMemcpy( this -> data,
                   rhs_vector. getData(),
                   this -> getSize() * sizeof( ElementType ),
                   cudaMemcpyDeviceToDevice ) != cudaSuccess )
   {
      checkCUDAError( __FILE__, __LINE__ );
      cerr << "Transfer of data on the CUDA device from ( " << rhs_vector. getName()
           << " ) to ( " << this -> getName() << " ) failed." << endl;
      return *this;
   }
   if( ! this -> getSafeMode() )
      cudaThreadSynchronize();
   return *this;
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
   return *this;
#endif
};

template< typename ElementType, typename IndexType >
void tnlArrayManager< ElementType, tnlCuda, IndexType > :: setElement( IndexType i, ElementType d )
{
   tnlAssert( this -> size != 0,
              cerr << "Vector name is " << this -> getName() );
   tnlAssert( i < this -> size,
            cerr << "You try to set non-existing element of the following vector."
                 << "Name: " << this -> getName() << endl
                 << "Size: " << this -> size << endl
                 << "Element number: " << i << endl; );
#ifdef HAVE_CUDA
   if( cudaMemcpy( ( void* ) &( this -> data[ i ] ),
                   ( void* ) &d,
                   sizeof( ElementType ),
                   cudaMemcpyHostToDevice ) != cudaSuccess )
   {
      cerr << "Transfer of data to the element number " << i
           << " of the CUDA long vector " << this -> getName()
           << " from the host failed." << endl;
      checkCUDAError( __FILE__, __LINE__ );
   }
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
   return;
#endif
};

template< typename ElementType, typename IndexType >
ElementType tnlArrayManager< ElementType, tnlCuda, IndexType > :: getElement( IndexType i ) const
{
   tnlAssert( this -> size != 0,
              cerr << "Vector name is " << this -> getName() );
   tnlAssert( i < this -> size,
            cerr << "You try to get non-existing element of the following vector."
                 << "Name: " << this -> getName() << endl
                 << "Size: " << this -> size << endl
                 << "Element number: " << i << endl; );
#ifdef HAVE_CUDA
   ElementType result;
   if( cudaMemcpy( ( void* ) &result,
                   ( void* )&( this -> data[ i ] ),
                   sizeof( ElementType ),
                   cudaMemcpyDeviceToHost ) != cudaSuccess )
   {
      cerr << "Transfer of data from the element number " << i
           << " of the CUDA long vector " << this -> getName()
           << " to the host failed." << endl;
      checkCUDAError( __FILE__, __LINE__ );
   }
   return result;
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
   return ElementType();
#endif
};

template< typename ElementType, typename IndexType >
ElementType& tnlArrayManager< ElementType, tnlCuda, IndexType > :: operator[] ( IndexType i )
{
   cerr << "I am sorry. You try to call the following operator: " << endl;
   cerr << endl;
   cerr << "ElementType& tnlArrayManager< ElementType, tnlCuda, IndexType > :: operator[] ( IndexType i )" << endl;
   cerr << endl;
   cerr << "for vector " << this -> getName() << "." << endl;
   cerr << "The call comes from the HOST (CPU) and the long vector is allocated " << endl;
   cerr << "on the CUDA device. Therefore we cannot return reference pointing to the" << endl;
   cerr << "different memory space. You may use the method getElement or setElement" << endl;
   cerr << "which are however very slow. You may also write specialised kernel for your task." << endl;
   abort();
}

template< typename ElementType, typename IndexType >
const ElementType& tnlArrayManager< ElementType, tnlCuda, IndexType > :: operator[] ( IndexType i ) const
{
   cerr << "I am sorry. You try to call the following operator: " << endl;
   cerr << endl;
   cerr << "ElementType& tnlArrayManager< ElementType, tnlCuda, IndexType > :: operator[] ( IndexType i )" << endl;
   cerr << endl;
   cerr << "for vector " << this -> getName() << "." << endl;
   cerr << "The call comes from the HOST (CPU) and the long vector is allocated " << endl;
   cerr << "on the CUDA device. Therefore we cannot return reference poointing to the" << endl;
   cerr << "different memory space. You may use the method getElement or setElement" << endl;
   cerr << "which are however very slow. You may also write specialised kernel for your task." << endl;
   abort();
}

template< typename ElementType, typename IndexType >
bool tnlArrayManager< ElementType, tnlCuda, IndexType > :: save( tnlFile& file ) const
{
   tnlAssert( this -> size != 0,
              cerr << "You try to save empty vector. Its name is " << this -> getName() );
   if( ! tnlObject :: save( file ) )
      return false;
   if( ! file. write( &this -> size, 1 ) )
      return false;
   if( ! file. write< ElementType, tnlCuda, IndexType >( this -> data, this -> size ) )
   {
      cerr << "I was not able to WRITE the long vector " << this -> getName()
           << " with size " << this -> getSize() << endl;
      return false;
   }
   //cerr << "Writing " << this -> size << " elements from " << this -> getName() << "." << endl;
   return true;
}

template< typename ElementType, typename IndexType >
bool tnlArrayManager< ElementType, tnlCuda, IndexType > :: load( tnlFile& file )
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
   if( ! file. read< ElementType, tnlCuda, IndexType >( this -> data, this -> size ) )
   {
      cerr << "I was not able to READ the long vector " << this -> getName()
           << " with size " << this -> getSize() << endl;
      return false;
   }
   return true;
}

template< typename Real, typename Index >
tnlArrayManager< Real, tnlCuda, Index > :: ~tnlArrayManager()
{
   dbgFunctionName( "tnlVectorCUDA", "~tnlVectorCUDA" );
#ifdef HAVE_CUDA
   if( this -> data && ! this -> shared_data )
   {
      dbgCout( "Freeing allocated memory of " << this -> getName() << " on CUDA device." );
      if( cudaFree( this -> data ) != cudaSuccess )
      {
         cerr << "Unable to free allocated memory on CUDA device of " << this -> getName() << "." << endl;
         checkCUDAError( __FILE__, __LINE__ );
      }
   }
#else
   //cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
};

template< typename ElementType, typename IndexType >
ostream& operator << ( ostream& str, const tnlArrayManager< ElementType, tnlCuda, IndexType >& vec )
{
#ifdef HAVE_CUDA
   IndexType size = vec. getSize();
   ElementType buffer[ 1024 ];
   IndexType pos( 0 );
   while( pos < size )
   {
      int transfer = Min( size - pos, 1024 );
      if( cudaMemcpy( buffer,
                      &( vec. getData()[ pos ] ),
                      transfer * sizeof( ElementType ), cudaMemcpyDeviceToHost ) != cudaSuccess )
      {
         cerr << "Transfer of data from CUDA device ( " << vec. getName()
                 << " ) failed." << endl;
         checkCUDAError( __FILE__, __LINE__ );
         return str;
      }
      for( IndexType i = 0; i < transfer; i ++ )
         str << buffer[ i ] <<  " ";
      pos += transfer;
   }
#else
   cerr << "CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
   return str;

};


#endif /* TNLARRAYMANAGERCUDA_H_ */
