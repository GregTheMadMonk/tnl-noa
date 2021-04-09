#include <iostream>
#include <array>
#include <TNL/Algorithms/staticFor.h>

int main( int argc, char* argv[] )
{
   // initiate std::array
   std::array< int, 5 > a{ 1, 2, 3, 4, 5 };

   // print out the array using template parameters for indexing
   TNL::Algorithms::staticFor< int, 0, 5 >(
      [&a] ( auto i ) {
         std::cout << "a[ " << i << " ] = " << std::get< i >( a ) << std::endl;
      }
   );
}
