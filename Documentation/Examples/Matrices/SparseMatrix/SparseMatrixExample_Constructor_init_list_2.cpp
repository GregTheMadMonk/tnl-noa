#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>


template< typename Device >
void initializerListExample()
{
   TNL::Matrices::SparseMatrix< double, Device > matrix ( 5, 5, {
      {  0,  0,  2.0 },
      {  1,  0, -1.0 }, {  1,  1,  2.0 }, {  1,  2, -1.0 },
      {  2,  1, -1.0 }, {  2,  2,  2.0 }, {  2,  3, -1.0 },
      {  3,  2, -1.0 }, {  3,  3,  2.0 }, {  3,  4, -1.0 },
      {  4,  4,  2.0 } } );

   std::cout << "General sparse matrix: " << std::endl << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Creating matrices on CPU ... " << std::endl;
   initializerListExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Creating matrices on CUDA GPU ... " << std::endl;
   initializerListExample< TNL::Devices::Cuda >();
#endif
}
