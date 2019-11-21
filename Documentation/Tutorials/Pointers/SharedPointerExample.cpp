#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Array.h>
#include <TNL/Pointers/SharedPointer.h>

using namespace TNL;

using ArrayCuda = Containers::Array< int, Devices::Cuda >;

struct Tuple
{
   Pointers::SharedPointer< ArrayCuda > a1, a2;
};

__global__ void checkArray( const Tuple t )
{
   printf( "Array size is: %d\n", ptr->getSize() );
   for( int i = 0; i < ptr->getSize(); i++ )
      printf( "a[ %d ] = %d \n", i, ( *ptr )[ i ] );
}

int main( int argc, char* argv[] )
{
   /***
    * Create a tuple of arrays and print the in CUDA kernel
    */
#ifdef HAVE_CUDA
   Tuple t;
   t.a1.modifyData< Devices::Host >().setSize( 10 );
   t.a1.modifyData< Devices::Host >() = 1;
   t.a2.modifyData< Devices::Host >().setSize( 10 );
   t.a2.modifyData< Devices::Host >() = 2;
   Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
   printkArrays<<< 1, 1 >>>( t );

   /***
    * Resize the array
    */
   t.a1.modifyData< Devices::Host >().setSize( 5 );
   t.a1.modifyData< Devices::Host >() = 3;
   t.a2.modifyData< Devices::Host >().setSize( 5 );
   t.a2.modifyData< Devices::Host >() = 4;
   Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
   printArrays<<< 1, 1 >>>( t );
#endif
   return EXIT_SUCCESS;

}

