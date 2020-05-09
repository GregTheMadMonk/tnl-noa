#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void getRowExample()
{
   TNL::Matrices::DenseMatrix< double, Device > matrix( 5, 5 );

   /***
    * We need a matrix view to pass the matrix to lambda function even on CUDA device.
    */
   auto matrixView = matrix.getView();
   auto f = [=] __cuda_callable__ ( int rowIdx ) mutable {
      auto row = matrixView.getRow( rowIdx );
      row.setElement( rowIdx, 10* ( rowIdx + 1 ) );
   };

   TNL::Algorithms::ParallelFor< Device >::exec( 0, matrix.getRows(), f );
   std::cout << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Getting matrix rows on host: " << std::endl;
   getRowExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Getting matrix rows on CUDA device: " << std::endl;
   getRowExample< TNL::Devices::Cuda >();
#endif
}