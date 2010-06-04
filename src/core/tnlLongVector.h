/***************************************************************************
                          tnlLongVector.h  -  description
                             -------------------
    begin                : 2007/06/16
    copyright            : (C) 2007 by Tomas Oberhuber
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

#ifndef tnlLongVectorH
#define tnlLongVectorH

#include <assert.h>
#include <string.h>
#include <core/tnlObject.h>
#include <core/param-types.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
void tnlLongVectorCUDASetValue( int* data,
                                const int size,
                                const int& v );

void tnlLongVectorCUDASetValue( float* data,
                                const int size,
                                const float& v );

void tnlLongVectorCUDASetValue( double* data,
                                const int size,
                                const double& v );

#else
#include <iostream>
using namespace std;
#endif


enum tnlDevice { tnlHost, tnlCuda };

template< class T, tnlDevice device > class tnlLongVectorCUDA;


template< class T, tnlDevice device = tnlHost > class tnlLongVector : public tnlObject
{

   public:

	bool setNewSize( int _size );

    bool setNewSize( const tnlLongVector< T, device >& v )
    {
    	return setNewSize( v. GetSize() );
    };

   //! Constructor with given size
   tnlLongVector( const char* name = 0, int _size = 0 )
   : size( _size ), shared_data( false ), data( 0 )
   {
      if( name )
         SetName( name );
      setNewSize( _size );
   };

   //! Constructor with another long vector as template
   tnlLongVector( const tnlLongVector& v )
   : tnlObject( v ), size( v. size ), shared_data( false ), data( 0 )
   {
	  setNewSize( v );
   };
   
   tnlLongVector( const tnlLongVectorCUDA< T, device >& v );

   tnlString getType() const;

   void setSharedData( T* _data, const int _size );

   int getSize() const
   {
      return size;
   };

   //! Returns pointer to data
   /*! This is not clear from the OOP point of view however it is necessary for keeping 
       good performance of derived numerical structure like solvers.
    */
   // TODO: return zero pointer if size == 0
   const T* Data() const
   {
      return data;
   };

   //! Returns pointer to data
   T* Data()
   {
      return data;
   }
  
   operator bool() const
   {
      return ( GetSize() != 0 );
   };
   
   T& operator[] ( int i );
   
   const T& operator[] ( int i ) const;

   void Zeros(); // TODO: replace by setValue

   void setValue( const T& v );

   bool copyFrom( const tnlLongVector< T >& long_vector );

   bool copyFrom( const tnlLongVectorCUDA< T, device >& cuda_vector );

   virtual ~tnlLongVector();

   //! Method for saving the object to a file as a binary data
   bool Save( ostream& file ) const;

   //! Method for restoring the object from a file
   bool Load( istream& file );

   protected:

   int size;

   T* data;

   bool shared_data;
};

template< class T, tnlDevice device > class tnlLongVector< T, tnlHost >
{
    bool setNewSize( int _size )
	{
    	if( tnlLongVector< T, device > :: size == _size ) return true;
    	if( data && ! shared_data )
    	{
    		delete[] -- data;
    		data = 0;
    	}
    	size = _size;
    	data = new T[ size + 1 ];
    	shared_data = false;
    	if( ! data )
    	{
    		cerr << "Unable to allocate new long vector with size " << size << "." << endl;
    		size = 0;
    		return false;
    	}
    	data ++;
    	return true;
	};

    tnlString getType() const
    {
       T t;
       return tnlString( "tnlLongVector< " ) + tnlString( GetParameterType( t ) ) + tnlString( ", tnlHost >" );
    };

    void setSharedData( T* _data, const int _size )
    {
       if( data && ! shared_data ) delete -- data;
       data = _data;
       shared_data = true;
       size = _size;
    };

    T& operator[] ( int i )
    {
       assert( i < size );
       return data[ i ];
    };

    const T& operator[] ( int i ) const
    {
       assert( i < size );
       return data[ i ];
    };

    void Zeros() // TODO: replace by setValue
    {
       int i;
       for( i = 0; i < size; i ++ ) data[ i ] = ( T ) 0;
    };

    void setValue( const T& v )
    {
 	   int i;
 	   for( i = 0; i < size; i ++ ) data[ i ] = ( T ) 0;
    };

    bool copyFrom( const tnlLongVector< T >& long_vector )
    {
       assert( long_vector. GetSize() == GetSize() );
       memcpy( data, long_vector. Data(), GetSize() * sizeof( T ) );
       return true;
    };

    bool copyFrom( const tnlLongVectorCUDA< T, device >& cuda_vector );

    virtual ~tnlLongVector()
    {
       if( data && ! shared_data ) delete -- data;
    };

    //! Method for saving the object to a file as a binary data
    bool Save( ostream& file ) const
    {
       if( ! tnlObject :: Save( file ) ) return false;
       file. write( ( char* ) &size, sizeof( int ) );
       if( file. bad() ) return false;
       file. write( ( char* ) data, size * sizeof( T ) );
       if( file. bad() ) return false;
       return true;
    };

    //! Method for restoring the object from a file
    bool Load( istream& file )
    {
       if( ! tnlObject :: Load( file ) ) return false;
       int _size;
       file. read( ( char* ) &_size, sizeof( int ) );
       if( _size <= 0 )
       {
          cerr << "Error: The size " << _size << " of the file is not a positive number." << endl;
          return false;
       }
       if( file. bad() ) return false;
       if( _size != size )
       {
          size = _size;
          if( ! shared_data ) delete[] --data;
          data = new T[ size + 1 ];
          if( ! data )
          {
             cerr << "Unable to allocate new long vector with size " << size << "." << endl;
             return false;
          }
          data ++;
       }
       file. read( ( char* ) data, size * sizeof( T ) );
       if( file. bad() ) return false;
       return true;
    };
};

template< class T, tnlDevice device > class tnlLongVector< T, tnlCuda >
{
    bool setNewSize( int _size )
	#ifdef HAVE_CUDA
	   {
	      dbgFunctionName( "tnlLongVectorCUDA", "setNewSize" );
	      dbgCout( "Setting new size to " << _size << " for " << GetName() );
	      if( size == _size ) return true;
	      if( data && ! shared_data )
	      {
	    	 dbgCout( "Freeing allocated memory on CUDA device of " << GetName() );
	         cudaFree( data );
	         data = NULL;
	      }
	      size = _size;
	      shared_data = false;
	      if( cudaMalloc( ( void** ) &data, size * sizeof( T ) ) != cudaSuccess )
	      {
	         cerr << "Unable to allocate new long vector with size "
	              << size * sizeof( T ) << " on CUDA device for "
	              << GetName() << "." << endl;
	         data = NULL;
	         size = 0;
	         return false;
	      }
	      return true;
	   };
	#else
	   {
	      cerr << "CUDA support is missing on this system." << endl;
	      return false;
	   };
	#endif

    tnlString getType() const
    {
       T t;
       return tnlString( "tnlLongVector< " ) + tnlString( GetParameterType( t ) ) + tnlString( ", tnlCuda >" );
    };

    void setSharedData( T* _data, const int _size )
 #ifdef HAVE_CUDA
    {
       if( data && ! shared_data ) cudaFree( data );
       data = _data;
       shared_data = true;
       size = _size;
    };
 #else
    {
       cerr << "CUDA support is missing on this system." << endl;
    };
 #endif

    bool copyFrom( const tnlLongVector< T, tnlHost >& long_vector )
    {
 #ifdef HAVE_CUDA
       assert( long_vector. GetSize() == GetSize() );
       if( cudaMemcpy( data, long_vector. Data(), GetSize() * sizeof( T ), cudaMemcpyHostToDevice ) != cudaSuccess )
       {
          cerr << "Transfer of data from CUDA host ( " << long_vector. GetName()
               << " ) to CUDA device ( " << GetName() << " ) failed." << endl;
          return false;
       }
       return true;
 #else
       cerr << "CUDA support is missing on this system." << endl;
       return false;
 #endif
    };

    bool copyFrom( const tnlLongVectorCUDA< T >& long_vector, bool safe_mod = true )
    {
 #ifdef HAVE_CUDA
       assert( long_vector. GetSize() == GetSize() );
       if( cudaMemcpy( data, long_vector. Data(), GetSize() * sizeof( T ), cudaMemcpyDeviceToDevice ) != cudaSuccess )
       {
          cerr << "Transfer of data from CUDA host ( " << long_vector. GetName()
               << " ) to CUDA device ( " << GetName() << " ) failed." << endl;
          return false;
       }
       if( safe_mod )
          cudaThreadSynchronize();
       return true;
 #else
       cerr << "CUDA support is missing on this system." << endl;
       return false;
 #endif
    };

    void setValue( const T& c )
    {
 #ifdef HAVE_CUDA
       :: tnlLongVectorCUDASetValue( this -> Data(), this -> GetSize(), c );
 #else
       cerr << "CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
 #endif
    };

    virtual
    ~tnlLongVectorCUDA()
    {
 	   dbgFunctionName( "tnlLongVectorCUDA", "~tnlLOngVectorCUDA" );
 #ifdef HAVE_CUDA
       if( data && ! shared_data )
       {
          dbgCout( "Freeing allocated memory of " << GetName() << " on CUDA device." );
          if( cudaFree( data ) != cudaSuccess )
          {
             cerr << "Unable to free alocated memory on CUDA device of " << GetName() << "." << endl;
          }
       }
 #else
       cerr << "CUDA support is missing on this system." << endl;
 #endif
    };

};



#include <core/tnlLongVectorCUDA.h>

template< typename T, tnlDevice device > tnlLongVector< T, device > :: tnlLongVector( const tnlLongVectorCUDA< T, device >& v )
#ifdef HAVE_CUDA
         : tnlObject( v ), size( v. GetSize() ), shared_data( false )
{
   data = new T[ size + 1 ];
   if( ! data )
   {
      cerr << "Unable to allocate new long vector with size " << size << "." << endl;
      abort();
   }
   data ++;
}
#else
{
   cerr << "CUDA is not supported on this system " << __FILE__ << " line " << __LINE__ << endl;
}
#endif

template< typename T, tnlDevice device > ostream& operator << ( ostream& o, const tnlLongVector< T, device >& v )
{
   int size = v. GetSize();
   int i;
   for( i = 0; i < size - 1; i ++ )
      o << v[ i ] << ",";
   o << v[ size - 1 ];
   return o;
};

template< typename T, tnlDevice device > void Copy( const tnlLongVector< T, device >& v1,
                                  tnlLongVector< T >& v2 )
{
   assert( v1. GetSize() == v2. GetSize() );
   memcpy( v2. Data(), v1. Data(), v1. GetSize() * sizeof( T ) );
};

// Explicit instatiation
template class tnlLongVector< double >;
template ostream& operator << ( ostream& o, const tnlLongVector< double >& v );


#endif
