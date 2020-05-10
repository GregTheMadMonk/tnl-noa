#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void createMatrixView()
{
   TNL::Containers::Vector< double, Device > values {
      1,  2,  3,  4,
      5,  6,  7,  8,
      9, 10, 11, 12 };

   TNL::Matrices::DenseMatrixView< double, Device, int, TNL::Containers::Segments::RowMajorOrder > matrix( 5, 5, values.getView() );

   /***
    * We need a matrix view to pass the matrix to lambda function even on CUDA device.
    */
   /*auto matrixView = matrix.getView();
   auto f = [=] __cuda_callable__ ( int rowIdx ) mutable {
      auto row = matrixView.getRow( rowIdx );
      row.setElement( rowIdx, 10* ( rowIdx + 1 ) );
   };

   TNL::Algorithms::ParallelFor< Device >::exec( 0, matrix.getRows(), f );
   std::cout << matrix << std::endl;*/
}

int main( int argc, char* argv[] )
{
   std::cout << "Creating matrix view on host: " << std::endl;
   createMatrixView< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Creating matrix view on CUDA device: " << std::endl;
   createMatrixView< TNL::Devices::Cuda >();
#endif
}
