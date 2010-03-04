/***************************************************************************
 tnlLongVectorCUDA.h  -  description
 -------------------
 begin                : Dec 27, 2009
 copyright            : (C) 2009 by Tomas Oberhuber
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

#ifndef TNLLONGVECTORCUDA_H_
#define TNLLONGVECTORCUDA_H_


/*
 *
 */
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

#include <core/tnlObject.h>
#include <core/param-types.h>
#include <core/tnlLongVector.h>
#include <debug/tnlDebug.h>

template< class T > class tnlLongVectorCUDA : public tnlObject
{
   public:
   //! Constructor with given size
   tnlLongVectorCUDA( const char* name = 0, int _size = 0 )
#ifdef HAVE_CUDA
    : tnlObject( name ), size( 0 ), data( NULL ), shared_data( false )
   {
      if( _size ) setNewSize( _size );
   };
#else
   {
      cerr << "CUDA support is missing in this system." << endl;
   }
#endif

   //! Constructor with another long vector as template
   tnlLongVectorCUDA( const tnlLongVectorCUDA& v )
#ifdef HAVE_CUDA
    : tnlObject( v ), size( 0 ), data( NULL ), shared_data( false )
   {
	  if( v. GetSize() )
		  setNewSize( v. size );
   };
#else
   {
      cerr << "CUDA support is missing on this system." << endl;
   }
#endif

   //! Constructor with another long vector as template
   tnlLongVectorCUDA( const tnlLongVector< T >& v )
#ifdef HAVE_CUDA
    : tnlObject( v ), size( 0 ), data( NULL ), shared_data( false )
   {
	   if( v. GetSize() ) setNewSize( v. GetSize() );
   };
#else
   {
      cerr << "CUDA support is missing on this system." << endl;
   }
#endif


   tnlString GetType() const
   {
      T t;
      return tnlString( "tnlLongVectorCUDA< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
   }


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

   bool setNewSize( const tnlLongVectorCUDA< T >& v )
   {
      return setNewSize( v. GetSize() );
   };

   bool setNewSize( const tnlLongVector< T >& v )
   {
	   return setNewSize( v. GetSize() );
   };

   void SetSharedData( T* _data, const int _size )
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

   int GetSize() const
   {
      return size;
   };

   //! Returns pointer to data
   /*! This is not clear from the OOP point of view however it is necessary for keeping
       good performance of derived numerical structure like solvers.
    */
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

   bool copyFrom( const tnlLongVector< T >& long_vector )
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

   private:

   int size;

   T* data;

   bool shared_data;

   static bool debug;

   //friend class tnlLongVectorCUDATester< T >;
};

template< class T > bool tnlLongVector< T > :: copyFrom( const tnlLongVectorCUDA< T >& cuda_vector )
{
#ifdef HAVE_CUDA
   assert( cuda_vector. GetSize() == GetSize() );
   if( cudaMemcpy( data, cuda_vector. Data(), GetSize() * sizeof( T ), cudaMemcpyDeviceToHost ) != cudaSuccess )
   {
      cerr << "Transfer of data from CUDA device ( " << cuda_vector. GetName()
           << " ) to CUDA host ( " << GetName() << " ) failed." << endl;
      return false;
   }
   return true;
#else
   cerr << "CUDA support is missing in this system." << endl;
   return false;
#endif

}

#endif /* TNLLONGVECTORCUDA_H_ */
