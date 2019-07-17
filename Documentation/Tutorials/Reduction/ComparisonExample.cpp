#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Algorithms/Reduction.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Containers::Algorithms;

template< typename Device >
bool comparison( const Vector< double, Device >& u, const Vector< double, Device >& v )
{
   auto u_view = u.getView();
   auto v_view = v.getView();

   /***
    * Fetch compares corresponding elements of both vectors
    */
   auto fetch = [=] __cuda_callable__ ( int i )->bool { return ( u_view[ i ] == v_view[ i ] ); };

   /***
    * Reduce performs logical AND on intermediate results obtained by fetch.
    */
   auto reduce = [] __cuda_callable__ ( bool& a, const bool& b ) { a = ( a && b ); };
   auto volatileReduce = [=] __cuda_callable__ ( volatile bool& a, const volatile bool& b ) { a = ( a && b ); };
   return Reduction< Device >::reduce( v_view.getSize(), reduce, volatileReduce, fetch, true );
}

int main( int argc, char* argv[] )
{
   Vector< double, Devices::Host > host_u( 10 ), host_v( 10 );
   host_u = 1.0;
   host_v.evaluate( [] __cuda_callable__ ( int i )->double { return 2 * ( i % 2 ) - 1; } );
   std::cout << "host_u = " << host_u << std::endl;
   std::cout << "host_v = " << host_v << std::endl;
   std::cout << "Comparison of host_u and host_v is: " << ( comparison( host_u, host_v ) ? "'true'" : "'false'" ) << "." << std::endl;
   std::cout << "Comparison of host_u and host_u is: " << ( comparison( host_u, host_u ) ? "'true'" : "'false'" ) << "." << std::endl;
#ifdef HAVE_CUDA
   Vector< double, Devices::Cuda > cuda_u( 10 ), cuda_v( 10 );
   cuda_u = 1.0;
   cuda_v.evaluate( [] __cuda_callable__ ( int i )->double { return 2 * ( i % 2 ) - 1; } );
   std::cout << "cuda_u = " << cuda_u << std::endl;
   std::cout << "cuda_v = " << cuda_v << std::endl;
   std::cout << "Comparison of cuda_u and cuda_v is: " << ( comparison( cuda_u, cuda_v ) ? "'true'" : "'false'" ) << "." << std::endl;
   std::cout << "Comparison of cuda_u and cuda_u is: " << ( comparison( cuda_u, cuda_u ) ? "'true'" : "'false'" ) << "." << std::endl;
#endif
   return EXIT_SUCCESS;
}

