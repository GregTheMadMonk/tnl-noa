#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void setElements()
{
   TNL::Matrices::DenseMatrix< double, Device > matrix( 5, 5 );
   for( int i = 0; i < 5; i++ )
      matrix.setElement( i, i, i );

   std::cout << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Set elements on host:" << std::endl;
   setElements< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Set elements on CUDA device:" << std::endl;
   setElements< TNL::Devices::Cuda >();
#endif
}
