#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Algorithms/Reduction.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Containers::Algorithms;

template< typename Device >
double maximumNorm( const Vector< double, Device >& v )
{
   auto view = v.getView();
   auto fetch = [=] __cuda_callable__ ( int i ) { return abs( view[ i ] ); };
   auto reduce = [] __cuda_callable__ ( double& a, const double& b ) { a = max( a, b ); };
   auto volatileReduce = [=] __cuda_callable__ ( volatile double& a, const volatile double& b ) { a = max( a ,b ); };
   return Reduction< Device >::reduce( v.getSize(), reduce, volatileReduce, fetch, 0.0 );
}

int main( int argc, char* argv[] )
{
   Vector< double, Devices::Host > host_v( 10 );
   host_v.evaluate( [] __cuda_callable__ ( int i )->double { return i - 7; } );
   std::cout << "host_v = " << host_v << std::cout;
   std::cout << "The maximum norm of the host vector elements is " << maximumNorm( host_v ) << "." << std::endl;
#ifdef HAVE_CUDA
   Vector< double, Devices::Cuda > cuda_v( 10 );
   cuda_v.evaluate( [] __cuda_callable__ ( int i )->double { return i - 7; } );
   std::cout << "cuda_v = " << cuda_v << std::cout;
   std::cout << "The maximum norm of the CUDA vector elements is " << maximumNorm( cuda_v ) << "." << std::endl;
#endif
   return EXIT_SUCCESS;
}

