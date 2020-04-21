#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>

int main( int argc, char* argv[] )
{
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Host > matrix {
      {  1,  2,  3,  4,  5,  6 },
      {  7,  8,  9, 10, 11, 12 },
      { 13, 14, 15, 16, 17, 18 }
   };

   std::cout << matrix << std::endl;

   TNL::Matrices::DenseMatrix< double, TNL::Devices::Host > triangularMatrix {
      {  1 },
      {  2,  3 },
      {  4,  5,  6 },
      {  7,  8,  9, 10 },
      { 11, 12, 13, 14, 15 }
   };

   std::cout << triangularMatrix << std::endl;
}
