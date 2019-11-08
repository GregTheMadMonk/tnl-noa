#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

template< typename Device >
void segmentedScan( Vector< double, Device >& v, Vector< bool, Device >& flags )
{
   /***
    * Reduction is sum of two numbers.
    */
   auto reduce = [] __cuda_callable__ ( const double& a, const double& b ) { return a + b; };

   /***
    * As parameters, we pass vector on which the scan is to be performed, interval
    * where the scan is performed, lambda function which is used by the scan and
    * zero element (idempotent) of the 'sum' operation.
    */
   SegmentedScan< Device >::perform( v, flags, 0, v.getSize(), reduce, 0.0 );
}

int main( int argc, char* argv[] )
{
   /***
    * Firstly, test the segmented prefix sum with vectors allocated on CPU.
    */
   Vector< bool, Devices::Host > host_flags{ 1,0,0,1,0,0,0,1,0,1,0,0, 0, 0 };
   Vector< double, Devices::Host > host_v { 1,3,5,2,4,6,9,3,5,3,6,9,12,15 };
   std::cout << "host_flags = " << host_flags << std::endl;
   std::cout << "host_v     = " << host_v << std::endl;
   segmentedScan( host_v, host_flags );
   std::cout << "The segmented prefix sum of the host vector is " << host_v << "." << std::endl;

   /***
    * And then also on GPU.
    */
#ifdef HAVE_CUDA
   //Vector< bool, Devices::Cuda > cuda_flags{ 1,0,0,1,0,0,0,1,0,1,0,0, 0, 0 };
   //Vector< double, Devices::Cuda > cuda_v { 1,3,5,2,4,6,9,3,5,3,6,9,12,15 };
   //std::cout << "cuda_flags = " << cuda_flags << std::endl;
   //std::cout << "cuda_v     = " << cuda_v << std::endl;
   //segmentedScan( cuda_v, cuda_flags );
   //std::cout << "The segmnted prefix sum of the CUDA vector is " << cuda_v << "." << std::endl;
#endif
   return EXIT_SUCCESS;
}

