__global void setZeros( float* A )
{
   int i = threadId. x;
   A[ i ] = 0.0;
}

void tnlLongVectorCUDATester< float > :: testKernel()
{
}