#include <tnlLongVectorCUDATester.cu.h>

void testKernelStarter( const float& number, const int size )
{
   testKernel( number, size );  
}

void testKernelStarter( const double& number, const int size )
{
  // testKernel( number, size );
}

