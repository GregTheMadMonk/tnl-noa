#include <iostream>
#include <TNL/Matrices/LambdaMatrix.h>

int main( int argc, char* argv[] )
{
   /***
    * Lambda functions defining the matrix.
    */
   auto rowLengths = [=] __cuda_callable__ ( const int rows, const int columns, const int rowIdx ) -> int { return 1; };
   auto matrixElements1 = [=] __cuda_callable__ ( const int rows, const int columns, const int rowIdx, const int localIdx, int& columnIdx, double& value ) {
         columnIdx = rowIdx;
         value =  1.0;
   };
   auto matrixElements2 = [=] __cuda_callable__ ( const int rows, const int columns, const int rowIdx, const int localIdx, int& columnIdx, double& value ) {
         columnIdx = rowIdx;
         value =  rowIdx;
   };

   const int size = 5;

   /***
    * Matrix construction with explicit type definition.
    */
   using MatrixType = decltype( TNL::Matrices::LambdaMatrixFactory< double, TNL::Devices::AnyDevice, int >::create( matrixElements1, rowLengths ) );
   MatrixType m1( size, size, matrixElements1, rowLengths );

   /***
    * Matrix construction using 'auto'.
    */
   auto m2 = TNL::Matrices::LambdaMatrixFactory< double, TNL::Devices::AnyDevice, int >::create( matrixElements2, rowLengths );
   m2.setDimensions( size, size );

   std::cout << "The first lambda matrix: " << std::endl << m1 << std::endl;
   std::cout << "The second lambda matrix: " << std::endl << m2 << std::endl;
}
