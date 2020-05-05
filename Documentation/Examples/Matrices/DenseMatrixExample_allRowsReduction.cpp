#include <iostream>
#include <iomanip>
#include <functional>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void allRowsReduction()
{
   TNL::Matrices::DenseMatrix< double, Device > matrix {
      {  1,  0,  0,  0,  0 },
      {  1,  2,  0,  0,  0 },
      {  0,  1,  8,  0,  0 },
      {  0,  0,  1,  9,  0 },
      {  0,  0,  0,  0,  1 } };

   /***
    * Find largest element in each row.
    */
   TNL::Containers::Vector< double, Device > rowMax( matrix.getRows() );

   /***
    * Prepare vector view and matrix view for lambdas.
    */
   const auto matrixView = matrix.getConstView();
   auto rowMaxView = rowMax.getView();

   /***
    * Fetch lambda just returns absolute value of matrix elements.
    */
   auto fetch = [=] __cuda_callable__ ( int rowIdx, int columnIdx, const double& value ) -> double {
      return TNL::abs( value );
   };

   /***
    * Reduce lambda return maximum of given values.
    */
   auto reduce = [=] __cuda_callable__ ( double& a, const double& b ) -> double {
      return TNL::max( a, b );
   };

   /***
    * Keep lambda store the largest value in each row to the vector rowMax.
    */
   auto keep = [=] __cuda_callable__ ( int rowIdx, const double& value ) mutable {
      rowMaxView[ rowIdx ] = value;
   };

   /***
    * Compute the largest values in each row.
    */
   matrix.allRowsReduction( fetch, reduce, keep, std::numeric_limits< double >::lowest() );

   std::cout << "Max. elements in rows are: " << rowMax << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "All rows reduction on host:" << std::endl;
   allRowsReduction< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "All rows reduction on CUDA device:" << std::endl;
   allRowsReduction< TNL::Devices::Cuda >();
#endif
}
