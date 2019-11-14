#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Array.h>

using namespace TNL;
using namespace TNL::Containers;

int main( int argc, char* argv[] )
{
   /****
    * Create an array on the host and print it on a console.
    */
   Array< int > host_array{ 1, 2, 3 };
   std::cout << "host_array = " << host_array << std::endl;

   return EXIT_SUCCESS;
}


