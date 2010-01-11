#include <cuda.h>
#include <cppunit/extensions/HelperMacros.h>
#include <core/tnlString.h>
#include <core/tnlLongVector.h>
#include <core/tnlLongVectorCUDA.h>
#include <core/tnlLongVectorCUDATester.h>

#ifndef tnlLongVectorCUDATester_cu_h
#define tnlLongVectorCUDATester_cu_h

//template< class T >
__global__ void setNumber( float* A, float c )
{
   int i = threadIdx. x;
   A[ i ] = c;
};

//template< class T >
void testKernel( float number, const int size )
{
   cout << number << " -- " << size << endl;
   /*tnlLongVectorCUDA< float > device_vector( size );
   tnlLongVector< float > host_vector( size );
   float* data = device_vector. Data();

   setNumber<<< 1, size >>>( data, number );
   host_vector. copyFrom( device_vector );*/

   float *h_a, *d_a;
   cudaMalloc( ( void** ) &d_a, size * sizeof( float ) );
   h_a = ( float* ) malloc( size * sizeof( float ) );
   setNumber<<< 1, size >>>( d_a, number );
   cudaMemcpy( h_a, d_a, size * sizeof( float ), cudaMemcpyDeviceToHost );

   int errors( 0 );
   for( int i = 0; i < size; i ++ )
   {
      cout << h_a[ i ] << "-";
      if( h_a[ i ] != number ) errors ++;
   }
   CPPUNIT_ASSERT( ! errors );
};

#endif
