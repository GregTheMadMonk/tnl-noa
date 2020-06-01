#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/MultidiagonalMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>


template< typename Device >
void laplaceOperatorMatrix()
{
   const int gridSize( 4 );
   const int matrixSize = gridSize * gridSize;
   TNL::Matrices::MultidiagonalMatrix< double, Device > matrix( 
      matrixSize, { - gridSize, -1, 0, 1, gridSize }, {
         {  0.0,  0.0, 1.0 },
         {  0.0,  0.0, 1.0 },
         {  0.0,  0.0, 1.0 },
         {  0.0,  0.0, 1.0 },
         {  0.0,  0.0, 1.0 },
         { -1.0, -1.0, 4.0, -1.0, -1.0 },
         { -1.0, -1.0, 4.0, -1.0, -1.0 },
         {  0.0,  0.0, 1.0 },
         {  0.0,  0.0, 1.0 },
         { -1.0, -1.0, 4.0, -1.0, -1.0 },
         { -1.0, -1.0, 4.0, -1.0, -1.0 },
         {  0.0,  0.0, 1.0 },
         {  0.0,  0.0, 1.0 },
         {  0.0,  0.0, 1.0 },
         {  0.0,  0.0, 1.0 },
         {  0.0,  0.0, 1.0 }
      } );
   std::cout << "Laplace operator matrix: " << std::endl << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Creating Laplace operator matrix on CPU ... " << std::endl;
   laplaceOperatorMatrix< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Creating Laplace operator matrix on CUDA GPU ... " << std::endl;
   laplaceOperatorMatrix< TNL::Devices::Cuda >();
#endif
}
