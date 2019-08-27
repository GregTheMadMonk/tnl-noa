#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/Algorithms/Reduction.h>

#include <TNL/Containers/StaticVector.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Containers::Algorithms;

template< typename Device >
void scan( const Vector< double, Device >& v )
{
   /****
    * Get vector view which can be captured by lambda.
    */
   auto view = v.getConstView();

   /****
    * The fetch function just reads elements of vector v.
    */
   auto fetch = [=] __cuda_callable__ ( int i ) -> double { return view[ i ]; };

   /***
    * Reduction is sum of two numbers.
    */
   auto reduce = [] __cuda_callable__ ( const double& a, const double& b ) { return a + b; };

   /***
    * Finally we call the templated function Reduction and pass number of elements to reduce,
    * lambdas defined above and finally value of idempotent element, zero in this case, which serve for the
    * reduction initiation.
    */
   Scan< Device >::perform( view, 0, view.getSize(), reduce, 0.0 );
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
   std::cout << "The prefisx sum of the CUDA vector is " << cuda_v << "." << std::endl;
#endif
   return EXIT_SUCCESS;
}

