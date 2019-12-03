#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/ParallelFor.h>

using namespace TNL;
using namespace TNL::Containers;

template< typename Device >
void meshFunctionSum( const int xSize,
                      const int ySize,
                      const Vector< double, Device >& v1,
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
   auto sum = [=] __cuda_callable__  ( int i, int j, const int xSize, const double c ) mutable {
      const int idx = j * xSize + i;
      result_view[ idx ] = v1_view[ idx ] + v2_view[ idx ] + c; };

   Algorithms::ParallelFor2D< Device >::exec( 0, 0, xSize, ySize, sum, xSize, c );
}

int main( int argc, char* argv[] )
{
   /***
    * Define dimensions of 2D mesh function.
    */
   const int xSize( 10 ), ySize( 10 );
   const int size = xSize * ySize;

   /***
    * Firstly, test the mesh functions sum on CPU.
    */
   Vector< double, Devices::Host > host_v1( size ), host_v2( size ), host_result( size );
   host_v1 = 1.0;
   host_v2 = 2.0;
   meshFunctionSum( xSize, ySize, host_v1, host_v2, 2.0, host_result );

   /***
    * And then also on GPU.
    */
#ifdef HAVE_CUDA
   Vector< double, Devices::Cuda > cuda_v1( size ), cuda_v2( size ), cuda_result( size );
   cuda_v1 = 1.0;
   cuda_v2 = 2.0;
   meshFunctionSum( xSize, ySize, cuda_v1, cuda_v2, 2.0, cuda_result );
#endif
   return EXIT_SUCCESS;
}

