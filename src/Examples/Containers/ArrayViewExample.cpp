#include <iostream>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/ArrayView.h>

using namespace TNL;

/***
 * The following works for any device (CPU, GPU ...).
 */
template< typename Device >
void arrayViewExample()
{
   const int size = 10;
   using ArrayType = Containers::Array< int, Device >;
   using IndexType = typename ArrayType::IndexType;
   using ViewType = Containers::ArrayView< int, Device >;
   ArrayType _a1( size ), _a2( size );
   ViewType a1 = _a1.getView();
   ViewType a2 = _a2.getView();

   /***
    * You may initiate the array view using setElement
    */
   for( int i = 0; i < size; i++ )
      a1.setElement( i, i );

   /***
    * You may also assign value to all array view elements ...
    */
   a2 = 0;

   /***
    * Simple array view values checks can be done as follows ...
    */
   if( a1.containsValue( 1 ) )
      std::cout << "a1 contains value 1." << std::endl;
   if( a1.containsValue( size ) )
      std::cout << "a1 contains value " << size << "." << std::endl;
   if( a1.containsOnlyValue( 0 ) )
      std::cout << "a2 contains only value 0." << std::endl;

   /***
    * More efficient way of array view elements manipulation is with the lambda functions
    */
   ArrayType _a3( size );
   ViewType a3 = _a3.getView();
   auto f1 = [] __cuda_callable__ ( IndexType i ) -> int { return 2 * i; };
   a3.evaluate( f1 );

   for( int i = 0; i < size; i++ )
      if( a3.getElement( i ) != 2 * i )
         std::cerr << "Something is wrong!!!" << std::endl;

   /***
    * You may swap array view data with the swap method.
    */
   a1.swap( a3 );

   /***
    * Of course, you may save it to file and load again
    */
   a1.save( "a1.tnl" );
   a2.load( "a1.tnl" );
   std::remove( "a1.tnl" );

   if( a2 != a1 )
      std::cerr << "Something is wrong!!!" << std::endl;

   std::cout << "a2 = " << a2 << std::endl;
}

int main()
{
   std::cout << "The first test runs on CPU ..." << std::endl;
   arrayViewExample< Devices::Host >();
#ifdef HAVE_CUDA
   std::cout << "The second test runs on GPU ..." << std::endl;
   arrayViewExample< Devices::Cuda >();
#endif
}
