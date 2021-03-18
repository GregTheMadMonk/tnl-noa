#include <iostream>
#include <TNL/Matrices/LambdaMatrix.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>


template< typename Device >
void forRowsExample()
{
   /**
    * Prepare lambda matrix of the following form:
    *
    * /  1   0   0   0   0 \
    * | -2   1  -2   0   0 |
    * |  0  -2   1  -2   0 |
    * |  0   0  -2   1  -2 |
    * |  0   0   0  -2   1 |
    * \  0   0   0   0   1 /
    */

   int size = 5;
   auto rowLengths = [=] __cuda_callable__ ( const int rows, const int columns, const int rowIdx ) -> int {
      if( rowIdx == 0 || rowIdx == size - 1 )
         return 1;
      return 3;
   };

   auto matrixElements = [=] __cuda_callable__ ( const int rows, const int columns, const int rowIdx, const int localIdx, int& columnIdx, double& value ) {
      if( rowIdx == 0 || rowIdx == size -1 )
      {
         columnIdx = rowIdx;
         value =  1.0;
      }
      else
      {
         columnIdx = rowIdx + localIdx - 1;
         value = ( columnIdx == rowIdx ) ? -2.0 : 1.0;
      }
   };

   using MatrixType = decltype( TNL::Matrices::LambdaMatrixFactory< double, Device, int >::create( matrixElements, rowLengths ) );
   MatrixType matrix( size, size, matrixElements, rowLengths );

   /**
    * Use the `forRows` method to copy the matrix elements to a dense matrix.
    */
   TNL::Matrices::DenseMatrix< double, Device, int > denseMatrix( size, size );
   denseMatrix.setValue( 0.0 );
   auto dense_view = denseMatrix.getView();
   auto f = [=] __cuda_callable__ ( const typename MatrixType::RowViewType& row ) mutable {
      auto dense_row = dense_view.getRow( row.getRowIndex() );
      for( int localIdx = 0; localIdx < row.getSize(); localIdx++ )
         dense_row.setElement( row.getColumnIndex( localIdx ), row.getValue( localIdx ) );
   };
   matrix.forAllRows( f );

   std::cout << "Lambda matrix looks as:" << std::endl << matrix << std::endl;
   std::cout << "Dense matrix looks as:" << std::endl << denseMatrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Running example on CPU ... " << std::endl;
   forRowsExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Running example on CUDA GPU ... " << std::endl;
   forRowsExample< TNL::Devices::Cuda >();
#endif
}
