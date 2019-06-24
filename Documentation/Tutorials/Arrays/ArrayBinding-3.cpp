#include <iostream>
#include <TNL/Containers/Array.h>

using namespace TNL;
using namespace TNL::Containers;

int main( int argc, char* argv[] )
{
   /****
    * Allocate data for all degrees of freedom
    */
   const int size = 5;
   Array< float > a( 3 * size );

   /***
    * Partition the data into density and velocity components
    */
   Array< float > rho( a,        0, size );
   Array< float > v_1( a,     size, size );
   Array< float > v_2( a, 2 * size, size );

   rho = 10.0;
   v_1 = 1.0;
   v_2 = 0.0;

   /****
    * Print the initialized arrays
    */
   std::cout << "rho =  " << rho << std::endl;
   std::cout << "v1 =  " << v_1 << std::endl;
   std::cout << "v2 =  " << v_2 << std::endl;
   std::cout << "a =  " << a << std::endl;
}
