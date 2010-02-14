#include <cuda.h>
#include <cppunit/extensions/HelperMacros.h>
#include <core/tnlString.h>
#include <core/tnlLongVector.h>
#include <core/tnlLongVectorCUDA.h>
#include <core/tnlLongVectorCUDATester.h>

#ifndef tnlLongVectorCUDATester_cu_h
#define tnlLongVectorCUDATester_cu_h

template< class T >
__global__ void setMultiBlockNumber( const T c, T* A, const int size )
{
   int i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size ) A[ i ] = c;
};

template< class T >
__global__ void setNumber( const T c, T* A, const int size )
{
   int i = threadIdx. x;
   if( i < size )
      A[ i ] = c;
};


template< class T >
void testMultiBlockKernel( const T& number, const int size )
{
   tnlLongVectorCUDA< T > device_vector( "device-vector", size );
   tnlLongVector< T > host_vector( "host-vector", size );
   T* data = device_vector. Data();

   const int block_size = 512;
   const int grid_size = size / 512 + 1;

   setMultiBlockNumber<<< grid_size, block_size >>>( number, data, size );
   host_vector. copyFrom( device_vector );

   int errors( 0 );
   for( int i = 0; i < size; i ++ )
   {
      //cout << host_vector[ i ] << "-";
      if( host_vector[ i ] != number ) errors ++;
   }
   CPPUNIT_ASSERT( ! errors );
};

template< class T >
void testKernel( const T& number, const int size )
{
   tnlLongVectorCUDA< T > device_vector( "device-vector", size );
   tnlLongVector< T > host_vector( "host-vector", size );
   T* data = device_vector. Data();
   setNumber<<< 1, size >>>( number, data, size );
   host_vector. copyFrom( device_vector );

   int errors( 0 );
   for( int i = 0; i < size; i ++ )
   {
      //cout << host_vector[ i ] << "-";
      if( host_vector[ i ] != number ) errors ++;
   }
   CPPUNIT_ASSERT( ! errors );
};

#endif
