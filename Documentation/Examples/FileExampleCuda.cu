#include <iostream>
#include <TNL/File.h>
#include <cuda.h>

using namespace TNL;
using namespace std;

int main()
{
   const int size = 3;
   double doubleArray[] = {  3.1415926535897932384626433,
                             2.7182818284590452353602874,
                             1.6180339887498948482045868 };

   /***
    * Save array to file.
    */
   File file;
   file.open( "test-file.tnl", std::ios_base::out | std::ios_base::trunc );
   file.save< double, double, Devices::Host >( doubleArray, size );
   file.close();

   /***
    * Allocate arrays on host and device
    */
   double *deviceArray, *hostArray;
   cudaMalloc( ( void** ) &deviceArray, size * sizeof( double ) );
   hostArray = new double[ 3 ];

   /***
    * Read array from the file to device
    */
   file.open( "test-file.tnl", std::ios_base::in );
   file.load< double, double, Devices::Cuda >( deviceArray, size );
   file.close();

   /***
    * Copy array from device to host
    */
   cudaMemcpy( ( void* ) hostArray, ( const void* ) deviceArray, size * sizeof( double), cudaMemcpyDeviceToHost );

   /***
    * Print the array on host
    */
   std::cout.precision( 15 );
   for( int i = 0; i < size; i++ )
      std::cout << hostArray[ i ] << std::endl;

   /***
    * Free allocated memory
    */
   cudaFree( deviceArray );
   delete[] hostArray;
}



