#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

template< typename Device >
void scan( Vector< double, Device >& v )
{
   /***
    * Reduction is sum of two numbers.
    */
   auto reduction = [] __cuda_callable__ ( const double& a, const double& b ) { return a + b; };

   /***
    * As parameters, we pass vector on which the scan is to be performed, interval
    * where the scan is performed, lambda function which is used by the scan and
    * zero element (idempotent) of the 'sum' operation.
    */
   Scan< Device >::perform( v, 0, v.getSize(), reduction, 0.0 );
}

int main( int argc, char* argv[] )
{
   /***
    * Firstly, test the prefix sum with vectors allocated on CPU.
    */
   Vector< double, Devices::Host > host_v( 10 );
   host_v = 1.0;
   std::cout << "host_v = " << host_v << std::endl;
   scan( host_v );
   std::cout << "The prefix sum of the host vector is " << host_v << "." << std::endl;

   /***
    * And then also on GPU.
    */
#ifdef HAVE_CUDA
   Vector< double, Devices::Cuda > cuda_v( 10 );
   cuda_v = 1.0;
   std::cout << "cuda_v = " << cuda_v << std::endl;
   scan( cuda_v );
   std::cout << "The prefix sum of the CUDA vector is " << cuda_v << "." << std::endl;
#endif
   return EXIT_SUCCESS;
}