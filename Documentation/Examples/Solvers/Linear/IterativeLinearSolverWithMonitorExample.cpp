#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Solvers/Linear/Jacobi.h>

template< typename Device >
void iterativeLinearSolverExample()
{
   /***
    * Set the following matrix (dots represent zero matrix elements):
    *
    *   /  2 -1    .    .    .   \
    *   |  -1    2 -1    .    .   |
    *   |  .    -1    2 -1.   .   |
    *   |  .    .    -1    2 -1   |
    *   \  .    .    .    -1    2 /
    */
   using MatrixType = TNL::Matrices::SparseMatrix< double, Device >;
   using Vector = TNL::Containers::Vector< double, Device >;
   const int size( 5 );
   auto matrix_ptr = std::make_shared< MatrixType >();
   matrix_ptr->setDimensions( size, size );
   matrix_ptr->setRowCapacities( Vector( { 2, 3, 3, 3, 2 } ) );

   auto f = [=] __cuda_callable__ ( typename MatrixType::RowView& row ) mutable {
      const int rowIdx = row.getRowIndex();
      if( rowIdx == 0 )
      {
         row.setElement( 0, rowIdx,    2.5 );    // diagonal element
         row.setElement( 1, rowIdx+1, -1 );      // element above the diagonal
      }
      else if( rowIdx == size - 1 )
      {
         row.setElement( 0, rowIdx-1, -1.0 );    // element below the diagonal
         row.setElement( 1, rowIdx,    2.5 );    // diagonal element
      }
      else
      {
         row.setElement( 0, rowIdx-1, -1.0 );    // element below the diagonal
         row.setElement( 1, rowIdx,    2.5 );    // diagonal element
         row.setElement( 2, rowIdx+1, -1.0 );    // element above the diagonal
      }
   };

   /***
    * Set the matrix elements.
    */
   matrix_ptr->forAllRows( f );
   std::cout << *matrix_ptr << std::endl;

   /***
    * Set the right-hand side vector
    */
   Vector x( size, 1.0 );
   Vector b( size );
   matrix_ptr->vectorProduct( x, b );
   x = 0.0;
   std::cout << "Vector b = " << b << std::endl;

   /***
    * Setup solver of the linear system
    */
   using LinearSolver = TNL::Solvers::Linear::Jacobi< MatrixType >;
   LinearSolver solver;
   solver.setMatrix( matrix_ptr );


   /***
    * Setup monitor of the iterative solver
    */
   using IterativeSolverMonitorType = TNL::Solvers::IterativeSolverMonitor< double, int >;
   IterativeSolverMonitorType monitor;
   TNL::Solvers::SolverMonitorThread t(monitor);
   monitor.setRefreshRate(100);  // refresh rate = timeout in milliseconds
   monitor.setVerbose(1);
   monitor.setStage( "Jacobi stage:" );
   TNL::Timer timer;
   monitor.setTimer( timer );
   solver.setSolverMonitor(monitor);
   solver.setOmega( 0.001 );
   //std::this_thread::sleep_for(std::chrono::milliseconds(3000)); // wait for 3 seconds to let the monitor doing something
   timer.start();
   solver.solve( b, x );
   monitor.stopMainLoop();
   std::cout << "Vector x = " << x << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Solving linear system on host: " << std::endl;
   iterativeLinearSolverExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Solving linear system on CUDA device: " << std::endl;
   iterativeLinearSolverExample< TNL::Devices::Cuda >();
#endif
}
