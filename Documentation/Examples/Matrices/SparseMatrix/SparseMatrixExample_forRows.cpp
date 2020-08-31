#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void forRowsExample()
{
   TNL::Matrices::SparseMatrix< double, Device > matrix( { 1, 2, 3, 4, 5 }, 5 );

   auto f = [=] __cuda_callable__ ( int rowIdx, int localIdx, int& columnIdx, double& value, bool& compute ) {
      if( rowIdx < localIdx )  // This is important, some matrix formats may allocate more matrix elements
                               // than we requested. These padding elements are processed here as well.
         compute = false;
      else
      {
         columnIdx = localIdx;
         value = rowIdx + localIdx + 1;
      }
   };

   matrix.forRows( 0, matrix.getRows(), f );
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
