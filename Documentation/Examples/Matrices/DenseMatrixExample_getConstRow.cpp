#include <iostream>
#include <functional>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>

int main( int argc, char* argv[] )
{
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Host > matrix {
      { 1, 0, 0, 0, 0 },
      { 1, 2, 0, 0, 0 },
      { 1, 2, 3, 0, 0 },
      { 1, 2, 3, 4, 0 },
      { 1, 2, 3, 4, 5 }
   };

   /***
    * We need a matrix view to pass the matrix to lambda function even on CUDA device.
    */
   const auto matrixView = matrix.getConstView();

   /***
    * Fetch lambda function returns diagonal element in each row.
    */
   auto fetch = [=] __cuda_callable__ ( int rowIdx ) mutable -> double {
      auto row = matrixView.getRow( rowIdx );
      return row.getElement( rowIdx );
   };

   int trace = TNL::Algorithms::Reduction< TNL::Devices::Host >::reduce( matrix.getRows(), std::plus<>{}, fetch, 0 );
   std::cout << "Matrix trace is " << trace << "." << std::endl;
}
