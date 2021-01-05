#include <iostream>
#include <TNL/Matrices/TridiagonalMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void forRowsExample()
{
   /***
    * Set the following matrix (dots represent zero matrix elements and zeros are
    * padding zeros for memory alignment):
    * 
    * 0 / 2  1  .  .  . \   -> { 0, 2, 1 }
    *   | 3  2  1  .  . |   -> { 3, 2, 1 }
    *   | .  3  2  1  . |   -> { 3, 2, 1 }
    *   | .  .  3  2  1 |   -> { 3, 2, 1 }
    *   \ .  .  .  3  2 / 0 -> { 3, 2, 0 } 
    */
   TNL::Matrices::TridiagonalMatrix< double, Device > matrix(
      5,      // number of matrix rows
      5 );    // number of matrix columns
   auto view = matrix.getView();

   auto f = [=] __cuda_callable__ ( int rowIdx, int localIdx, int columnIdx, double& value, bool& compute ) {
      /***
       * 'forRows' method iterates only over matrix elements lying on given subdiagonals
       * and so we do not need to check anything. The element value can be expressed
       * by the 'localIdx' variable, see the following figure:
       * 
       *                           0  1  2  <- localIdx values
       *                           -------
       * 0 / 2  1  .  .  . \   -> { 0, 2, 1 }
       *   | 3  2  1  .  . |   -> { 3, 2, 1 }
       *   | .  3  2  1  . |   -> { 3, 2, 1 }
       *   | .  .  3  2  1 |   -> { 3, 2, 1 }
       *   \ .  .  .  3  2 / 0 -> { 3, 2, 0 } 
       * 
       */
      value = 3 - localIdx;
   };
   view.forRows( 0, matrix.getRows(), f );
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
