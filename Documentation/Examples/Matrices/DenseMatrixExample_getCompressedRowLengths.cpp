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

   std::cout << triangularMatrix << std::endl;

   TNL::Containers::Vector< int, TNL::Devices::Host > rowLengths;
   triangularMatrix.getCompressedRowLengths( rowLengths );

   std::cout << "Compressed row lengths are: " << rowLengths << std::endl;
}
