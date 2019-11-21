#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Array.h>
#include <TNL/Pointers/UniquePointer.h>


using namespace TNL;

using ArrayHost = Containers::Array< int, Devices::Host >;
using ArrayCuda = Containers::Array< int, Devices::Cuda >;

__global__ void checkArray( const ArrayCuda* ptr )
{
   printf( "Array size is: %d\n", ptr->getSize() );
   for( int i = 0; i < ptr->getSize(); i++ )
      printf( "a[ %d ] = %d \n", i, ( *ptr )[ i ] );
}

int main( int argc, char* argv[] )
{

   /***
    * Make unique pointer on array on CPU and manipulate the
    * array via the pointer.
    */
   Pointers::UniquePointer< ArrayHost > array_host_ptr( 10 );
   *array_host_ptr = 1;
   std::cout << "Array = " << *array_host_ptr << std::endl;

   /***
    * Let's do the same in CUDA
    */
#ifdef HAVE_CUDA
   Pointers::UniquePointer< ArrayCuda > array_cuda_ptr( 10 );
   array_cuda_ptr.modifyData< Devices::Host >() = 1;
   //Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
   //checkArray<<< 1, 1 >>>( &array_cuda_ptr.getData< Devices::Cuda >() );
#endif
   return EXIT_SUCCESS;
}

