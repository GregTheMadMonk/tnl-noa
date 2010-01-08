#include <core/tnlLongVectorCUDATester.h>

__global__ void setZeros( float* A )
{
   int i = threadIdx. x;
   A[ i ] = 0.0;
}

