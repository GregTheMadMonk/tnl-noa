#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Algorithms/Reduction.h>
#include <TNL/Timer.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Containers::Algorithms;

template< typename Device >
class ReductionClass
{
   public:
      
      static double reduce()
      {
         using VectorType = Vector< double, Device >;
         using ViewType = VectorView< double, Device >;
         
         VectorType v( 100 );
         ViewType view = v.getView();
         
         auto fetch = [=] __cuda_callable__ ( int i ) { return view[ i ]; };
         auto reduce = [] __cuda_callable__ ( double& a, const double& b ) { a += b; };
         auto volatileReduce = [] __cuda_callable__ ( volatile double& a, const volatile double& b ) { a += b; };

         return Reduction< Device >::reduce( view.getSize(), reduce, volatileReduce, fetch, 0.0 );
         
      }
};

int main( int argc, char* argv[] )
{
   ReductionClass< Devices::Host >::reduce();
}