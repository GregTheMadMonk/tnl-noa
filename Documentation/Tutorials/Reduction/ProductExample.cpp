#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Algorithms/Reduction.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Containers::Algorithms;

template< typename Device >
double product( const Vector< double, Device >& v )
{
   auto view = v.getView();
   auto fetch = [=] __cuda_callable__ ( int i ) { return view[ i ]; };
   auto reduce = [] __cuda_callable__ ( double& a, const double& b ) { a *= b; };
   auto volatileReduce = [=] __cuda_callable__ ( volatile double& a, const volatile double& b ) { a *= b; };

   /***
    * Since we compute the product of all elements, the reduction must be initialized by 1.0 not by 0.0.
    */
   return Reduction< Device >::reduce( view.getSize(), reduce, volatileReduce, fetch, 1.0 );
}

int main( int argc, char* argv[] )
{
   /***
    * The first test on CPU ...
    */
   Vector< double, Devices::Host > host_v( 10 );
   host_v = 1.0;
   std::cout << "host_v = " << host_v << std::endl;
   std::cout << "The product of the host vector elements is " << product( host_v ) << "." << std::endl;

   /***
    * ... the second test on GPU.
    */
#ifdef HAVE_CUDA
   Vector< double, Devices::Cuda > cuda_v( 10 );
   cuda_v = 1.0;
   std::cout << "cuda_v = " << cuda_v << std::endl;
   std::cout << "The product of the CUDA vector elements is " << product( cuda_v ) << "." << std::endl;
#endif
   return EXIT_SUCCESS;
}

