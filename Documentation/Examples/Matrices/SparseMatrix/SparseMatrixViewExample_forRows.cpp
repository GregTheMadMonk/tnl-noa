#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void forRowsExample()
{
   using MatrixType = TNL::Matrices::SparseMatrix< double, Device >;
   using RowViewType = typename MatrixType::ViewType::RowViewType;
   MatrixType matrix( { 1, 2, 3, 4, 5, }, 5  );
   auto view = matrix.getView();

   auto f = [] __cuda_callable__ ( RowViewType& row ) mutable {
      for( int localIdx = 0;
           localIdx <= row.getRowIndex(); // This is important, some matrix formats may allocate more matrix elements
           localIdx++ )                   // than we requested. These padding elements are processed here as well.
                                          // and so we cannot use row.getSize()
      {
         row.setValue( localIdx, row.getRowIndex() - localIdx + 1.0 );
         row.setColumnIndex( localIdx, localIdx );
      }
   };
   view.forAllRows( f );

   std::cout << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Creating matrix on host: " << std::endl;
   forRowsExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Creating matrix on CUDA device: " << std::endl;
   forRowsExample< TNL::Devices::Cuda >();
#endif
}
