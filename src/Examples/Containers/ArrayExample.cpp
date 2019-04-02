#include <iostream>
#include <list>
#include <vector>
#include <TNL/Containers/Array.h>

using namespace TNL;
using namespace std;

/***
 * The following works for any device (CPU, GPU ...).
 */
template< typename Device >
void arrayExample()
{
   const int size = 10;
   using ArrayType = Containers::Array< int, Device >;
   using IndexType = typename ArrayType::IndexType;
   ArrayType a1( size ), a2( size );

   /***
    * You may initiate the array using setElement
    */
   for( int i = 0; i< size; i++ )
      a1.setElement( i, i );

   /***
    * You may also assign value to all array elements ...
    */
   a2 = 0.0;

   /***
    * ... or assign STL list and vector.
    */
   std::list< float > l = { 1.0, 2.0, 3.0 };
   std::vector< float > v = { 5.0, 6.0, 7.0 };
   a1 = v;
   a1 = l;
   
   /***
    * Simple array values checks can be done as follows ...
    */
   if( a1.containsValue( 1 ) )
      std::cout << "a1 contains value 1." << std::endl;
   if( a1.containsValue( size ) )
      std::cout << "a1 contains value " << size << "." << std::endl;
   if( a1.containsOnlyValue( 0 ) )
      std::cout << "a2 contains only value 0." << std::endl;

   /***
    * More efficient way of array elements manipulation is with the lambda functions
    */
   ArrayType a3( size );
   auto f1 = [] __cuda_callable__ ( IndexType i ) -> int { return 2 * i;};
   a3.evaluate( f1 );

   for( int i = 0; i < size; i++ )
      if( a3.getElement( i ) != 2 * i )
         std::cerr << "Something is wrong!!!" << std::endl;

   /***
    * You may swap array data with the swap method.
    */
   a1.swap( a3 );

   /***
    * Of course, you may save it to file and load again
    */
   a1.save( "a1.tnl" );
   a2.load( "a1.tnl" );

   if( a2 != a1 )
      std::cerr << "Something is wrong!!!" << std::endl;

   std::cout << "a2 = " << a2 << std::endl;
}

int main()
{
   std::cout << "The first test runs on CPU ..." << std::endl;
   arrayExample< Devices::Host >();
#ifdef HAVE_CUDA
   std::cout << "The second test runs on GPU ..." << std::endl;
   arrayExample< Devices::Cuda >();
#endif
}
