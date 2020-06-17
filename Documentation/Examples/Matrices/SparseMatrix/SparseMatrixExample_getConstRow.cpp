#include <iostream>
#include <functional>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Pointers/SharedPointer.h>

template< typename Device >
void getRowExample()
{
   using MatrixType = TNL::Matrices::SparseMatrix< double, Device >;
   TNL::Pointers::SharedPointer< MatrixType > matrix ( 5, 5 );
   matrix->setElements(
      {  { 0, 0, 1 },
         { 1, 0, 1 }, { 1, 1, 2 },
         { 2, 0, 1 }, { 2, 1, 2 }, { 2, 2, 3 },
         { 3, 0, 1 }, { 3, 1, 2 }, { 3, 2, 3 }, { 3, 3, 4 },
         { 4, 0, 1 }, { 4, 1, 2 }, { 4, 2, 3 }, { 4, 3, 4 }, { 4, 4, 5 } } );

   /***
    * Fetch lambda function returns diagonal element in each row.
    */
   auto fetch = [=] __cuda_callable__ ( int rowIdx ) mutable -> double {
      auto row = matrix->getRow( rowIdx );
      return row.getValue( rowIdx );
   };

   /***
    * For the case when Device is CUDA device we need to synchronize smart
    * pointers. To avoid this you may use SparseMatrixView. See
    * SparseMatrixView::getConstRow example for details.
    */
   TNL::Pointers::synchronizeSmartPointersOnDevice< Device >();

   /***
    * Compute the matrix trace.
    */
   int trace = TNL::Algorithms::Reduction< Device >::reduce( 0, matrix->getRows(), std::plus<>{}, fetch, 0 );
   std::cout << "Matrix trace is " << trace << "." << std::endl;
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
