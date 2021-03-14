#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void forElementsExample()
{
   TNL::Matrices::DenseMatrix< double, Device > matrix( 5, 5 );

   auto f = [=] __cuda_callable__ ( int rowIdx, int columnIdx, int columnIdx_, double& value, bool& compute ) {
      if( rowIdx < columnIdx )
         compute = false;
      else
         value = rowIdx + columnIdx;
   };

   matrix.forElements( 0, matrix.getRows(), f );
   std::cout << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Creating matrix on host: " << std::endl;
   forElementsExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Creating matrix on CUDA device: " << std::endl;
   forElementsExample< TNL::Devices::Cuda >();
#endif
}
