#include <iostream>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/ArrayView.h>

using namespace TNL;
using namespace TNL::Containers;

int main( int argc, char* argv[] )
{
   /****
    * Create new arrays and initiate them
    */
   const int size = 10;
   Array< float, Devices::Cuda > a( size ), b( size );
   a = 0;
   b.getView().evaluate( [=] __cuda_callable__ ( int i ) -> float { return i; } );

   /****
    * Test the values store in the arrays
    */
   if( a.containsValue( 0.0 ) )
      std::cout << "a contains 0" << std::endl;

   if( a.containsValue( 1.0 ) )
      std::cout << "a contains 1" << std::endl;

   if( b.containsValue( 0.0 ) )
      std::cout << "b contains 0" << std::endl;

   if( b.containsValue( 1.0 ) )
      std::cout << "b contains 1" << std::endl;

   if( a.containsOnlyValue( 0.0 ) )
      std::cout << "a contains only 0" << std::endl;

   if( a.containsOnlyValue( 1.0 ) )
      std::cout << "a contains only 1" << std::endl;

   if( b.containsOnlyValue( 0.0 ) )
      std::cout << "b contains only 0" << std::endl;

   if( b.containsOnlyValue( 1.0 ) )
      std::cout << "b contains only 1" << std::endl;

   /****
    * Change the first half of b and test it again
    */
   b.getView().evaluate( [=] __cuda_callable__ ( int i ) -> float { return 0.0; }, 0, 5 );
   if( b.containsOnlyValue( 0.0, 0, 5 ) )
      std::cout << "First five elements of b contains only 0" << std::endl;
}

