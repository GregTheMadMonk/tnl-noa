#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Algorithms/Reduction.h>
#include <TNL/Timer.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Containers::Algorithms;

template< typename Device >
double maskAndReduce( Vector< double, Device >& u )
{
   auto u_view = u.getView();
   auto fetch = [=] __cuda_callable__ ( int i )->double {
      if( i % 2 == 0 ) return u_view[ i ];
      return 0.0; };
   auto reduce = [] __cuda_callable__ ( double& a, const double& b ) { a += b; };
   auto volatileReduce = [=] __cuda_callable__ ( volatile double& a, const volatile double& b ) { a += b; };
   return Reduction< Device >::reduce( u_view.getSize(), reduce, volatileReduce, fetch, 0.0 );
}

int main( int argc, char* argv[] )
{
   Timer timer;
   Vector< double, Devices::Host > host_u( 100000 );
   host_u = 1.0;
   timer.start();
   double result = maskAndReduce( host_u );
   timer.stop();
   std::cout << "Host tesult is:" << result << ". It took " << timer.getRealTime() << "seconds." << std::endl;
#ifdef HAVE_CUDA
   Vector< double, Devices::Cuda > cuda_u( 100000 );
   cuda_u = 1.0;
   timer.reset();
   timer.start();
   result = maskAndReduce( cuda_u );
   timer.stop();
   std::cout << "CUDA result is:" << result << ". It took " << timer.getRealTime() << "seconds." << std::endl;
#endif
   return EXIT_SUCCESS;
}

