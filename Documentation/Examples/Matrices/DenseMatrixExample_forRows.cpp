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
   auto f = [=] __cuda_callable__ ( int rowIdx, int columnIdx, int globalIdx, double& value, bool& compute ) {
      if( rowIdx < columnIdx )
         compute = false;
      else
         value = rowIdx + columnIdx;
   };

   matrix.forRows( 0, matrix.getRows(), f );
   std::cout << matrix << std::endl;
}
