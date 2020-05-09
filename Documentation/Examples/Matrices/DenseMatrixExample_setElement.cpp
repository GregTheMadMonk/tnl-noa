#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Pointers/SmartPointersRegister.h>

template< typename Device >
void setElements()
{
   TNL::Pointers::SharedPointer< TNL::Matrices::DenseMatrix< double, Device > > matrix( 5, 5 );
   for( int i = 0; i < 5; i++ )
      matrix->setElement( i, i, i );

   std::cout << "Matrix set from the host:" << std::endl;
   std::cout << *matrix << std::endl;

   auto f = [=] __cuda_callable__ ( int i ) mutable {
      matrix->setElement( i, i, -i );
   };
   TNL::Pointers::synchronizeSmartPointersOnDevice< Device >();
   TNL::Algorithms::ParallelFor< Device >::exec( 0, 5, f );

   std::cout << "Matrix set from its native device:" << std::endl;
   std::cout << *matrix << std::endl;

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
