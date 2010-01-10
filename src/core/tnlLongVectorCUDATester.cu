#include <cuda.h> 
#include <core/tnlString.h>

#include <core/tnlLongVector.h>
#include <core/tnlLongVectorCUDA.h>
#include <core/tnlLongVectorCUDATester.h>

__global__ void setNumber( float* A, float c )
{
   int i = threadIdx. x;
   A[ i ] = c;
}

void testKernel()
{
   tnlLongVectorCUDA< float > device_vector( 500 );
   tnlLongVector< float > host_vector( 500 );
   float* data = device_vector. Data();
   setNumber<<< 1, 500 >>>( data, 0.0 );
   host_vector. copyFrom( device_vector );
   int errors( 0 );
   for( int i = 0; i < 500; i ++ )
   {
      if( host_vector[ i ] != 0.0 ) errors ++;
      cout << host_vector[ i ] << "-";
   }
   CPPUNIT_ASSERT( ! errors );
   setNumber<<< 1, 500 >>>( data, 1.0 );
   host_vector. copyFrom( device_vector );
   errors = 0;
   for( int i = 0; i < 500; i ++ )
   {
      if( host_vector[ i ] != 1.0 ) errors ++;
      cout << host_vector[ i ] << "-";
   }
   CPPUNIT_ASSERT( ! errors );
}