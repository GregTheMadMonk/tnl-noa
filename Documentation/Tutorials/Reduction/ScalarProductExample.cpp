#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Algorithms/Reduction.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Containers::Algorithms;

template< typename Device >
double scalarProduct( const Vector< double, Device >& u, const Vector< double, Device >& v )
{
   auto u_view = u.getView();
   auto v_view = v.getView();
   auto fetch = [=] __cuda_callable__ ( int i ) { return u_view[ i ] * v_view[ i ]; };
   auto reduce = [] __cuda_callable__ ( double& a, const double& b ) { a += b; };
   auto volatileReduce = [=] __cuda_callable__ ( volatile double& a, const volatile double& b ) { a += b; };
   return Reduction< Device >::reduce( v.getSize(), reduce, volatileReduce, fetch, 0.0 );
}

int main( int argc, char* argv[] )
{
   Vector< double, Devices::Host > host_u( 10 ), host_v( 10 );
   host_u = 1.0;
   host_v.evaluate( [] __cuda_callable__ ( int i )->double { return 2 * ( i % 2 ) - 1; } );
   std::cout << "host_u = " << host_u << std::cout;
   std::cout << "host_v = " << host_v << std::cout;
   std::cout << "The scalar product ( host_u, host_v ) is " << scalarProduct( host_u, host_v ) << "." << std::endl;
   std::cout << "The scalar product ( host_v, host_v ) is " << scalarProduct( host_v, host_v ) << "." << std::endl;
#ifdef HAVE_CUDA
   Vector< double, Devices::Cuda > cuda_u( 10 ), cuda_v( 10 );
   cuda_u = 1.0;
   cuda_v.evaluate( [] __cuda_callable__ ( int i )->double { return 2 * ( i % 2 ) - 1; } );
   std::cout << "cuda_u = " << cuda_u << std::cout;
   std::cout << "cuda_v = " << cuda_v << std::cout;
   std::cout << "The scalar product ( cuda_u, cuda_v ) is " << scalarProduct( cuda_u, cuda_v ) << "." << std::endl;
   std::cout << "The scalar product ( cuda_v, cuda_v ) is " << scalarProduct( cuda_v, cuda_v ) << "." << std::endl;
#endif
   return EXIT_SUCCESS;
}

