#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Reduction.h>
#include <TNL/Timer.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

template< typename Device >
double mapReduce( Vector< double, Device >& u )
{
   auto u_view = u.getView();
   auto fetch = [=] __cuda_callable__ ( int i )->double {
      return u_view[ 2 * i ]; };
   auto reduce = [] __cuda_callable__ ( const double& a, const double& b ) { return a + b; };
   return Reduction< Device >::reduce( u_view.getSize() / 2, reduce, fetch, 0.0 );
}

int main( int argc, char* argv[] )
{
   Timer timer;
   Vector< double, Devices::Host > host_u( 100000 );
   host_u = 1.0;
   timer.start();
   double result = mapReduce( host_u );
   timer.stop();
   std::cout << "Host result is:" << result << ". It took " << timer.getRealTime() << "seconds." << std::endl;
#ifdef HAVE_CUDA
   Vector< double, Devices::Cuda > cuda_u( 100000 );
   cuda_u = 1.0;
   timer.reset();
   timer.start();
   result = mapReduce( cuda_u );
   timer.stop();
   std::cout << "CUDA result is:" << result << ". It took " << timer.getRealTime() << "seconds." << std::endl;
#endif
   return EXIT_SUCCESS;
}

