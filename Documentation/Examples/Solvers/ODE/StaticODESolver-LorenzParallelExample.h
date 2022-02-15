#include <iostream>
#include <fstream>
#include <TNL/Solvers/ODE/StaticEuler.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/ParallelFor.h>

using Real = double;

template< typename Device >
void solveParallelODEs( const char* file_name )
{
   using Vector = TNL::Containers::StaticVector< 3, Real >;
   using ODESolver = TNL::Solvers::ODE::StaticEuler< Vector >;
   const Real final_t = 10.0;
   const Real tau = 0.001;
   const Real output_time_step = 0.01;
   const Real sigma_min( 0.0 ), rho_min( 0.0 ), beta_min( 0.0 );
   const int parametric_steps = 4;
   const Real parametric_step = 30.0 / ( parametric_steps - 1 );
   const int output_time_steps = final_t / output_time_step + 1;

   const int results_size( output_time_steps * parametric_steps * parametric_steps * parametric_steps );
   TNL::Containers::Vector< Vector, Device > results( results_size, 0.0 );
   auto results_view = results.getView();
   auto f = [=] __cuda_callable__ ( const Real& t, const Real& tau, const Vector& u, Vector& fu,
                                    const Real& sigma_i, const Real& rho_j, const Real& beta_k ) {
         const Real& x = u[ 0 ];
         const Real& y = u[ 1 ];
         const Real& z = u[ 2 ];
         fu[ 0 ] = sigma_i * (y - x );
         fu[ 1 ] = rho_j * x - y - x * z;
         fu[ 2 ] = -beta_k * z + x * y;
      };
   auto solve = [=] ( int i, int j, int k ) mutable {
      const Real sigma_i = sigma_min + i * parametric_step;
      const Real rho_j   = rho_min + j * parametric_step;
      const Real beta_k  = beta_min + k * parametric_step;

      ODESolver solver;
      solver.setTau(  tau );
      solver.setTime( 0.0 );
      Vector u( 1.0, 1.0, 1.0 );
      int time_step( 0 );
      results_view[ ( i * parametric_step + j ) * parametric_step + k ] = u;
      while( solver.getTime() < final_t )
      {
         solver.setStopTime( TNL::min( solver.getTime() + output_time_step, final_t ) );
         solver.solve( u, f, sigma_i, rho_j, beta_k );
         const int idx = ( ( time_step++ * parametric_steps + i ) * parametric_steps + j ) * parametric_steps + k;
         results_view[ idx ] = u;
      }
   };
   TNL::Algorithms::ParallelFor3D< Device >::exec( 0, 0, 0, parametric_steps, parametric_steps, parametric_steps, solve );

   std::fstream file;
   file.open( file_name, std::ios::out );
   for( int sigma_idx = 0; sigma_idx < parametric_steps; sigma_idx++ )
      for( int rho_idx = 0; rho_idx < parametric_steps; rho_idx++ )
         for( int beta_idx = 0; beta_idx < parametric_steps; beta_idx++ )
         {
            Real sigma = sigma_min + sigma_idx * parametric_step;
            Real rho   = rho_min   + rho_idx * parametric_step;
            Real beta  = beta_min  + beta_idx * parametric_step;
            file << "# sigma " << sigma << " rho " << rho << " beta " << beta << std::endl;
            for( int i = 0; i < output_time_steps; i++ )
            {
               int offset = ( ( i * parametric_steps + sigma_idx ) * parametric_steps + rho_idx ) * parametric_steps + beta_idx;
               auto u = results.getElement(  offset );
               file << u[ 0 ] << " " << u[  1 ] << " " << u[ 2 ] << std::endl;
            }
            file << std::endl;
         }
}

int main( int argc, char* argv[] )
{
   solveParallelODEs< TNL::Devices::Host >( "StaticODESolver-LorenzParallelExample-Host.out" );
#ifdef HAVE_CUDA
   solveParallelODEs< TNL::Devices::Cuda >( "StaticODESolver-LorenzParallelExample-Cuda.out" );
#endif
}
