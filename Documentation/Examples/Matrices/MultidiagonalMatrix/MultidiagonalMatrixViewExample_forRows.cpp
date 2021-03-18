#include <iostream>
#include <TNL/Matrices/MultidiagonalMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void forRowsExample()
{
   using MatrixType = TNL::Matrices::MultidiagonalMatrix< double, Device >;
   /***
    * Set the following matrix (dots represent zero matrix elements and zeros are
    * padding zeros for memory alignment):
    *
    *    0 /  1 -2  .  .  . \  -> { 0, 0, 1 }
    *      | -2  1 -2  .  . |  -> { 0, 2, 1 }
    *      |  . -2  1 -2. . |  -> { 3, 2, 1 }
    *      |  .  . -2  1 -2 |  -> { 3, 2, 1 }
    *      \  .  .  . -2  1 /  -> { 3, 2, 1 }
    *
    * The diagonals offsets are { -1, 0, 1 }.
    */

    const int size = 5;
    MatrixType matrix(
      size,            // number of matrix rows
      size,            // number of matrix columns
      { -2, -1, 0 } ); // matrix diagonals offsets
   auto view = matrix.getView();

   auto f = [=] __cuda_callable__ ( typename MatrixType::RowViewType& row ) {
      /***
       * 'forElements' method iterates only over matrix elements lying on given subdiagonals
       * and so we do not need to check anything. The element value can be expressed
       * by the 'localIdx' variable, see the following figure:
       *
       *                                0  1  2  <- localIdx values
       *                              ----------
       *    0 /  1 -2  .  .  . \  -> {  0, 1, -2 }
       *      | -2  1 -2  .  . |  -> { -2, 1, -2 }
       *      |  . -2  1 -2. . |  -> { -2, 1, -2 }
       *      |  .  . -2  1 -2 |  -> { -2, 1, -2 }
       *      \  .  .  . -2  1 /  -> { -2, 1,  0 }
       *
       */
      const int& rowIdx = row.getRowIndex();
      row.setElement( 1, 1.0 );
      if( rowIdx > 0 )
         row.setElement( 0, -2.0 );
      if( rowIdx < size - 1 )
         row.setElement( 2, -2.0 );
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
