#include <iostream>
#include <TNL/Solvers/ODE/StaticEuler.h>

using Real = double;

int main( int argc, char* argv[] )
{
   using ODESolver = TNL::Solvers::ODE::StaticEuler< Real >;
   const Real final_t = 10.0;
   const Real tau = 0.001;
   const Real output_time_step = 0.1;

   ODESolver solver;
   solver.setTau(  tau );
   solver.setTime( 0.0 );
   Real u = 0.0;
   while( solver.getTime() < final_t )
   {
      solver.setStopTime( TNL::min( solver.getTime() + output_time_step, final_t ) );
      solver.solve( u, [] ( const Real& t, const Real& tau, const Real& u, Real& fu ) {
         fu = sin( t ) * t;
      } );
      std::cout << solver.getTime() << " " << u << std::endl;
   }
}