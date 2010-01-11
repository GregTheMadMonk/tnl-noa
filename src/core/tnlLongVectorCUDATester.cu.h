#include <cuda.h>
#include <cppunit/extensions/HelperMacros.h>
#include <core/tnlString.h>
#include <core/tnlLongVector.h>
#include <core/tnlLongVectorCUDA.h>
#include <core/tnlLongVectorCUDATester.h>

#ifndef tnlLongVectorCUDATester_cu_h
#define tnlLongVectorCUDATester_cu_h

template< class T > __global__ void setNumber( T* A, T c )
{
   int i = threadIdx. x;
   A[ i ] = c;
};

template< class T > void testKernel( const T& number, const int size )
{
   tnlLongVectorCUDA< T > device_vector( size );
   tnlLongVector< T > host_vector( size );
   T* data = device_vector. Data();
   setNumber<<< 1, size >>>( data, number );
   host_vector. copyFrom( device_vector );
   int errors( 0 );
   for( int i = 0; i < size; i ++ )
   {
	   cout << host_vector[ i ] << "-";
       if( host_vector[ i ] != number ) errors ++;
   }
   CPPUNIT_ASSERT( ! errors );
};

#endif
