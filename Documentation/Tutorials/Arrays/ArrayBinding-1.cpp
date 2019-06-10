#include <iostream>
#include <TNL/Containers/Array.h>

using namespace TNL;
using namespace TNL::Containers;

int main( int argc, char* argv[] )
{
   /****
    * Allocate an array on host
    */
   const int size = 10;
   int* ai = new int[ size ];

   /****
    * Bind the data with TNL array
    */
   Array< int > host_array;
   host_array.bind( ai, size );

   /****
    * Initialize the data using the TNL array
    */
   host_array = 66;

   /****
    * Check the data
    */
   for( int i = 0; i < size; i++ )
      std::cout << i << " ";
   std::cout << std::endl;
}
