#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>

int main( int argc, char* argv[] )
{
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Host > triangularMatrix {
      {  1 },
      {  2,  3 },
      {  4,  5,  6 },
      {  7,  8,  9, 10 },
      { 11, 12, 13, 14, 15 }
   };

   std::cout << "Matrix elements count is " << triangularMatrix.getElementsCount() << "." << std::endl;
   std::cout << "Non-zero matrix elements count is " << triangularMatrix.getNonzeroElementsCount() << "." << std::endl;
}
