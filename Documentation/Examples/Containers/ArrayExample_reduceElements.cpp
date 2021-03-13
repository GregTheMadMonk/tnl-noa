#include <iostream>
#include <functional>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/ArrayView.h>

using namespace TNL;

template< typename Device >
void reduceElementsExample()
{
   /****
    * Create new arrays
    */
   const int size = 10;
   Containers::Array< float, Device > a( size );

   /****
    * Initiate the elements of array `a`
    */
   a.forEachElement( [] __cuda_callable__ ( int i, float& value ) { value = i; } );

   /****
    * Sum all elements of array `a`
    */
   auto fetch = [=] __cuda_callable__ ( int i, float& value ) { return value; };
   auto sum = a.reduceEachElement( fetch, std::plus<>{}, 0.0 );

   /****
    * Print the results
    */
   std::cout << " a = " << a << std::endl;
   std::cout << " sum = " << sum << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Running example on the host system: " << std::endl;
   reduceElementsExample< Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Running example on the CUDA device: " << std::endl;
   reduceElementsExample< Devices::Cuda >();
#endif
}
