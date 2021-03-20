#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void forRowsExample()
{
   using MatrixType = TNL::Matrices::DenseMatrix< double, Device >;
   using RowView = typename MatrixType::RowView;
   MatrixType matrix( 5, 5 );

   auto f = [=] __cuda_callable__ ( RowView& row ) mutable {
      const int& rowIdx = row.getRowIndex();
      row.setElement( rowIdx, 10 * ( rowIdx + 1 ) );
   };

   /***
    * Set the matrix elements.
    */
   matrix.forAllRows( f );
   std::cout << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Getting matrix rows on host: " << std::endl;
   forRowsExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Getting matrix rows on CUDA device: " << std::endl;
   forRowsExample< TNL::Devices::Cuda >();
#endif
}
