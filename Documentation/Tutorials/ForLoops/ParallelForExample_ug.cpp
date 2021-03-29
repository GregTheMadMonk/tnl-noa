#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/ParallelFor.h>

using namespace TNL;
using namespace TNL::Containers;

template< typename Device >
void vectorSum( const Vector< double, Device >& v1,
                const Vector< double, Device >& v2,
                const double& c,
                Vector< double, Device >& result )
{
   /****
    * Get vectors view which can be captured by lambda.
    */
   auto v1_view = v1.getConstView();
   auto v2_view = v2.getConstView();
   auto result_view = result.getView();

   /****
    * The sum function.
    */
   auto sum = [=] __cuda_callable__  ( int i, const double c ) mutable {
      result_view[ i ] = v1_view[ i ] + v2_view[ i ] + c; };

   Algorithms::ParallelFor< Device >::exec( 0, v1.getSize(), sum, c );
}

int main( int argc, char* argv[] )
{
   /***
    * Firstly, test the vectors sum on CPU.
    */
   Vector< double, Devices::Host > host_v1( 10 ), host_v2( 10 ), host_result( 10 );
   host_v1 = 1.0;
   host_v2.forAllElements( []__cuda_callable__ ( int i, double& value ) { value = i; } );
   vectorSum( host_v1, host_v2, 2.0, host_result );
   std::cout << "host_v1 = " << host_v1 << std::endl;
   std::cout << "host_v2 = " << host_v2 << std::endl;
   std::cout << "The sum of the vectors on CPU is " << host_result << "." << std::endl;

   /***
    * And then also on GPU.
    */
#ifdef HAVE_CUDA
   Vector< double, Devices::Cuda > cuda_v1( 10 ), cuda_v2( 10 ), cuda_result( 10 );
   cuda_v1 = 1.0;
   cuda_v2.forAllElements( []__cuda_callable__ ( int i, double& value ) { value = i; } );
   vectorSum( cuda_v1, cuda_v2, 2.0, cuda_result );
   std::cout << "cuda_v1 = " << cuda_v1 << std::endl;
   std::cout << "cuda_v2 = " << cuda_v2 << std::endl;
   std::cout << "The sum of the vectors on GPU is " << cuda_result << "." << std::endl;
#endif
   return EXIT_SUCCESS;
}

