#include <iostream>
#include <cstdlib>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Algorithms/StaticFor.h>

using namespace TNL;
using namespace TNL::Containers;

int main( int argc, char* argv[] )
{
   /****
    * Create two static vectors
    */
   const int Size( 3 );
   StaticVector< Size, double > a, b;
   a = 1.0;
   b = 2.0;
   double sum( 0.0 );

   /****
    * Compute an addition of a vector and a constant number.
    */
   auto addition = [&]( int i, const double& c ) { a[ i ] = b[ i ] + c; sum += a[ i ]; };
   Algorithms::StaticFor< 0, Size >::exec( addition, 3.14 );
   std::cout << "a = " << a << std::endl;
   std::cout << "sum = " << sum << std::endl;
}

