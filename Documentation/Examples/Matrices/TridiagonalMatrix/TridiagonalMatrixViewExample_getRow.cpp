#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/TridiagonalMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void getRowExample()
{
   const int matrixSize( 5 );
   using MatrixType = TNL::Matrices::TridiagonalMatrix< double, Device >;
   MatrixType matrix(
      matrixSize,  // number of matrix rows
      matrixSize  // number of matrix columns
   );
   auto view = matrix.getView();

   auto f = [=] __cuda_callable__ ( int rowIdx ) mutable {
      //auto row = matrix->getRow( rowIdx );    
      // For some reason the previous line of code is not accepted by nvcc 10.1 
      // so we replace it with the following two lines.
      //auto ref = matrix.modifyData();
      auto row = view.getRow( rowIdx );

      if( rowIdx > 0 )
         row.setElement( 0, -1.0 );  // elements below the diagonal
      row.setElement( 1, 2.0 );      // elements on the diagonal
      if( rowIdx < matrixSize - 1 )  // elements above the diagonal
         row.setElement( 2, -1.0 );
   };

   /***
    * Set the matrix elements.
    */
   TNL::Algorithms::ParallelFor< Device >::exec( 0, view.getRows(), f );
   std::cout << std::endl << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Getting matrix rows on host: " << std::endl;
   getRowExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   // It seems that nvcc 10.1 does not handle lambda functions properly. 
   // It is hard to make nvcc to compile this example and it does not work
   // properly. We will try it with later version of CUDA.
   std::cout << "Getting matrix rows on CUDA device: " << std::endl;
   getRowExample< TNL::Devices::Cuda >();
#endif
}
