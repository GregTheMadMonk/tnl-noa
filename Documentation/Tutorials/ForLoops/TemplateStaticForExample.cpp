#include <iostream>
#include <cstdlib>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Algorithms/TemplateStaticFor.h>

using namespace TNL;
using namespace TNL::Containers;

using Index = int;
const Index Size( 5 );

template< Index I >
struct LoopBody
{
   static void exec( const StaticVector< Size, double >& v ) {
      std::cout << "v[ " << I << " ] = " << v[ I ] << std::endl;
   }
};

int main( int argc, char* argv[] )
{
   /****
    * Initiate static vector
    */
   StaticVector< Size, double > v{ 1.0, 2.0, 3.0, 4.0, 5.0 };

   /****
    * Print out the vector using template parameters for indexing.
    */
   Algorithms::TemplateStaticFor< Index, 0, Size, LoopBody >::exec( v );
}

