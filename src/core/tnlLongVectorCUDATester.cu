#include <tnlLongVectorCUDATester.cu.h>

void testMultiBlockKernelStarter( const int& number, const int size )
{
   testMultiBlockKernel( number, size );  
}

void testMultiBlockKernelStarter( const float& number, const int size )
{
   testMultiBlockKernel( number, size );  
}

void testMultiBlockKernelStarter( const double& number, const int size )
{
   testMultiBlockKernel( number, size );
}

void testKernelStarter( const int& number, const int size )
{
   testKernel( number, size );  
}

void testKernelStarter( const float& number, const int size )
{
   testKernel( number, size );  
}

void testKernelStarter( const double& number, const int size )
{
   testKernel( number, size );
}

