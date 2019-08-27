#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Algorithms/Reduction.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Containers::Algorithms;

template< typename Device >
std::pair< int, double >
maximumNorm( const Vector< double, Device >& v )
{
   auto view = v.getConstView();

   auto fetch = [=] __cuda_callable__ ( int i ) { return abs( view[ i ] ); };
   auto reduction = [] __cuda_callable__ ( int& aIdx, const int& bIdx, double& a, const double& b ) {
      if( a < b ) {
         a = b;
         aIdx = bIdx;
      }
      else if( a == b && bIdx < aIdx )
         aIdx = bIdx;
   };
   return Reduction< Device >::reduceWithArgument( view.getSize(), reduction, fetch, std::numeric_limits< double >::max() );
}

int main( int argc, char* argv[] )
{
   Vector< double, Devices::Host > host_v( 10 );
   host_v.evaluate( [] __cuda_callable__ ( int i )->double { return i - 7; } );
   std::cout << "host_v = " << host_v << std::endl;
   auto maxNormHost = maximumNorm( host_v );
   std::cout << "The maximum norm of the host vector elements is " <<  maxNormHost.second << " at position " << maxNormHost.first << "." << std::endl;
#ifdef HAVE_CUDA
   Vector< double, Devices::Cuda > cuda_v( 10 );
   cuda_v.evaluate( [] __cuda_callable__ ( int i )->double { return i - 7; } );
   std::cout << "cuda_v = " << cuda_v << std::endl;
   auto maxNormCuda = maximumNorm( cuda_v );
   std::cout << "The maximum norm of the device vector elements is " <<  maxNormCuda.second << " at position " << maxNormCuda.first << "." << std::endl;
#endif
   return EXIT_SUCCESS;
}

