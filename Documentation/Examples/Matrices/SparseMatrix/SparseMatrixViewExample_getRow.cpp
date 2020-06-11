#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Pointers/SharedPointer.h>

template< typename Device >
void getRowExample()
{
   TNL::Matrices::SparseMatrix< double, Device > matrix( { 1, 1, 1, 1, 1 }, 5 );
   auto view = matrix.getView();

   auto f = [=] __cuda_callable__ ( int rowIdx ) mutable {
      auto row = view.getRow( rowIdx );
      row.setElement( 0, rowIdx, 10 * ( rowIdx + 1 ) );
   };

   /***
    * Set the matrix elements.
    */
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
