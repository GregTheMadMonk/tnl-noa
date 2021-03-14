#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>

using namespace TNL;
using namespace TNL::Containers;

template< typename Device >
void expressions()
{
   using RealType = float;
   using VectorType = Vector< RealType, Device >;
   using ViewType = VectorView< RealType, Device >;

   /****
    * Create vectors
    */
   const int size = 11;
   VectorType a_v( size ), b_v( size ), c_v( size );
   ViewType a = a_v.getView();
   ViewType b = b_v.getView();
   ViewType c = c_v.getView();
   a.forEachElement( [] __cuda_callable__ ( int i, RealType& value ) { value = 3.14 * ( i - 5.0 ) / 5.0; } );
   b = a * a;
   c = 3 * a + sign( a ) * sin( a );
   std::cout << "a = " << a << std::endl;
   std::cout << "sin( a ) = " << sin( a ) << std::endl;
   std::cout << "abs( sin( a ) ) = " << abs( sin ( a ) ) << std::endl;
   std::cout << "b = " <<  b << std::endl;
   std::cout << "c = " <<  c << std::endl;
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
   std::cout << std::endl;
   std::cout << "Expressions on GPU ..." << std::endl;
   expressions< Devices::Cuda >();
}


