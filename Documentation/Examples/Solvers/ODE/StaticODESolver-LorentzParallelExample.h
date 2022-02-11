#include <iostream>
#include <TNL/Solvers/ODE/StaticEuler.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/ParallelFor.h>

using Real = double;

template< typename Device >
void solveParallelODEs()
{
   using Vector = TNL::Containers::StaticVector< 3, Real >;
   using ODESolver = TNL::Solvers::ODE::StaticEuler< Vector >;
   const Real final_t = 10.0;
   const Real tau = 0.001;
   const Real output_time_step = 0.1;
   const Real sigma_min( 0.0 ), rho_min( 0.0 ), beta_min( 0.0 );
   const int parametric_steps = 11;
   const Real parametric_step = 30.0 / ( parametric_steps - 1 );
   const int time_steps = final_t / tau + 1;

   const int results_size( time_steps * parametric_steps * parametric_steps * parametric_steps );
   TNL::Containers::Vector< Vector, Device > results( results_size, 0.0 );
   auto results_view = results.getView();
   auto f = [=] __cuda_callable__ ( const Real& t, const Real& tau, const Vector& u, Vector& fu,
                                    const Real& sigma, const Real& rho, const Real& beta ) {
         const Real& x = u[ 0 ];
         const Real& y = u[ 1 ];
         const Real& z = u[ 2 ];
         fu[ 0 ] = sigma * (y - x );
         fu[ 1 ] = rho * x - y - x * z;
         fu[ 2 ] = -beta * z + x * y;
      };
   auto solve = [=] ( int i, int j, int k ) mutable {
      const Real sigma = sigma_min + i * parametric_step;
      const Real rho   = rho_min + j * parametric_step;
      const Real beta  = beta_min + k * parametric_step;

      ODESolver solver;
      solver.setTau(  tau );
      solver.setTime( 0.0 );
      Vector u = 0.0;
      int time_step( 0 );
      results_view[ ( i * parametric_step + j ) * parametric_step + k ] = u;
      while( solver.getTime() < final_t )
      {
         solver.setStopTime( TNL::min( solver.getTime() + output_time_step, final_t ) );
         solver.solve( u, f, sigma, rho, beta );
         const int idx = ( ( ++time_step * parametric_step + i ) * parametric_step + j ) * parametric_step + k;
         results_view[ idx ] = u;
      }
   };
   TNL::Algorithms::ParallelFor3D< Device >::exec( 0, 0, 0, parametric_steps, parametric_steps, parametric_steps, solve );
}

int main( int argc, char* argv[] )
{
   solveParallelODEs< TNL::Devices::Host >();
#ifdef HAVE_CUDA
   solveParallelODEs< TNL::Devices::Cuda >();
#endif
}
