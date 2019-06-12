#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>

using namespace TNL;
using namespace TNL::Containers;

template< typename Device >
void expressions()
{
   using VectorType = Vector< float, Device >;
   using ViewType = VectorView< float, Device >;
   /****
    * Create vectors
    */
   const int size = 6;
   VectorType a_v( size ), b_v( size ), c_v( size );
   ViewType a = a_v.getView();
   ViewType b = b_v.getView();
   ViewType c = c_v.getView();
   a.evaluate( [] __cuda_callable__ ( int i )->float { return i - 3;} );
   b = abs( a );
   c = sign( b );

   std::cout << "a = " << a << std::endl;
   std::cout << "b = " << b << std::endl;
   std::cout << "c = " << c << std::endl;
   std::cout << "a + 3 * b + c * min( c, 0 ) = " <<  a + 3 * b + c * min( c, 0 ) << std::endl;
}

int main( int argc, char* argv[] )
{
   /****
    * Perform test on CPU
    */
   std::cout << "Expressions on CPU ..." << std::endl;
   expressions< Devices::Host >();

   /****
    * Perform test on GPU
    */
   std::cout << "Expressions on GPU ..." << std::endl;
   expressions< Devices::Cuda >();
}


