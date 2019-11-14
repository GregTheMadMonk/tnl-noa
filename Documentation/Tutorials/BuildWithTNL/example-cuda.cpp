#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Array.h>
#include <TNL/Devices/Cuda.h>

using namespace TNL;
using namespace TNL::Containers;

int main( int argc, char* argv[] )
{
   /****
    * Create an array on the host and print it on a console.
    */
   Array< int > host_array{ 1, 2, 3 };
   std::cout << "host_array = " << host_array << std::endl;

   /****
    * Create another array on GPU and print it on a console as well.
    */
   Array< int, Devices::Cuda > device_array{ 4, 5, 6 };
   std::cout << "device_array = " << device_array << std::endl;
   return EXIT_SUCCESS;
}


