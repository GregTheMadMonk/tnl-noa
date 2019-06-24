#include <iostream>
#include <TNL/Containers/Array.h>

using namespace TNL;
using namespace TNL::Containers;

void initArray( Array< int >& a )
{
   /****
    * Create new array, bind it with 'a' and initialize it
    */
   Array< int > b( 10 );
   a.bind( b );
   b = 10;

   /****
    * Show that both arrays share the same data
    */
   std::cout << "a data in initArray function is " << a.getData() << std::endl;
   std::cout << "a value in initArray function is " << a << std::endl;
   std::cout << "--------------------------------------" << std::endl;
   std::cout << "b data in initArray function is " << b.getData() << std::endl;
   std::cout << "b in initArray function is " << b << std::endl;
   std::cout << "--------------------------------------" << std::endl;
}

int main( int argc, char* argv[] )
{
   /****
    * Create array but do not initialize it
    */
   Array< int > a;

   /***
    * Call function initArray for the array initialization
    */
   initArray( a );

   /****
    * Print the initialized array
    */
   std::cout << "a data in main function is " << a.getData() << std::endl;
   std::cout << "a in main function is " << a << std::endl;
}
