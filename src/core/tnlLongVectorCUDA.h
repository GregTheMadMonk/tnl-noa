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

#ifdef HAVE_CUDA

/*
 *
 */
#include <cuda_runtime.h>
#include <core/tnlObject.h>
#include <core/param-types.h>
#include <core/tnlLongVector.h>

template< class T > class tnlLongVectorCUDA : public tnlObject
{
   public:
   //! Constructor with given size
   tnlLongVectorCUDA( int _size = 0 )
    : size( _size ), shared_data( false )
   {
      cudaMalloc( ( void** ) &data, ( size + 1 ) * sizeof( T ) );
      if( ! data )
      {
         cerr << "Unable to allocate new long vector with size " << size << " on CUDA device." << endl;
         abort();
      }
      //data ++;
    };

   //! Constructor with another long vector as template
   tnlLongVectorCUDA( const tnlLongVectorCUDA& v )
    : tnlObject( v ), size( v. size ), shared_data( false )
   {
      cudaMalloc( ( void** ) &data, ( size + 1 ) * sizeof( T ) );
      if( ! data )
      {
         cerr << "Unable to allocate new long vector with size " << size << " on CUDA device." << endl;
         abort();
      }
      //data ++;
   };

   tnlString GetType() const
   {
      T t;
      return tnlString( "tnlLongVectorCUDA< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
   }

   bool SetNewSize( int _size )
   {
      if( size == _size ) return true;
      if( ! shared_data )
      {
         cudaFree( data );
         data = 0;
      }
      size = _size;
      cudaMalloc( ( void** ) &data, size * sizeof( T ) );
      shared_data = false;
      if( ! data )
      {
         cerr << "Unable to allocate new long vector with size " << size << " on CUDA device." << endl;
         size = 0;
         return false;
      }
      return true;
   };

   bool SetNewSize( const tnlLongVectorCUDA< T >& v )
   {
      return SetNewSize( v. GetSize() );
   };

   void SetSharedData( T* _data, const int _size )
   {
      if( data && ! shared_data ) cudaFree( data );
      data = _data;
      shared_data = true;
      size = _size;
   };

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
      assert( long_vector. GetSize() == GetSize() );
      cudaMemcpy( data, long_vector. Data(), GetSize() * sizeof( T ), cudaMemcpyHostToDevice );
      return true;
   }

   virtual
   ~tnlLongVectorCUDA()
   {
      if( data && ! shared_data ) cudaFree( data );
   };

   private:

   int size;

   T* data;

   bool shared_data;

   //friend class tnlLongVectorCUDATester< T >;
};

template< class T > bool tnlLongVector< T > :: copyFrom( const tnlLongVectorCUDA< T >& cuda_vector )
{
   assert( cuda_vector. GetSize() == GetSize() );
   cudaMemcpy( data, cuda_vector. Data(), GetSize() * sizeof( T ), cudaMemcpyDeviceToHost );
   return true;
}

#endif
#endif /* TNLLONGVECTORCUDA_H_ */
