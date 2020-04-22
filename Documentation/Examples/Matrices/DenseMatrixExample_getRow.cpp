#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>

int main( int argc, char* argv[] )
{
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Host > matrix( 5, 5 );

   /***
    * We need a matrix view to pass the matrix to lambda function even on CUDA device.
    */
   auto matrixView = matrix.getView();
   auto f = [=] __cuda_callable__ ( int rowIdx ) mutable {
      auto row = matrixView.getRow( rowIdx );
      row.setElement( rowIdx, 10* ( rowIdx + 1 ) );
   };

   TNL::Algorithms::ParallelFor< TNL::Devices::Host >::exec( 0, matrix.getRows(), f );
   std::cout << matrix << std::endl;
}
