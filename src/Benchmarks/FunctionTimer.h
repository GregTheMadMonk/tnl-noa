/***************************************************************************
                          FunctionTimer.h  -  description
                             -------------------
    begin                : Dec 25, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky,
//                 Tomas Oberhuber

#pragma once

#include <type_traits>

#include <TNL/Timer.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Solvers/IterativeSolverMonitor.h>

namespace TNL {
namespace Benchmarks {

template< typename Device >
class FunctionTimer
{
public:
   // returns a pair of (mean, stddev) where mean is the arithmetic mean of the
   // computation times and stddev is the sample standard deviation
   template< typename ComputeFunction,
             typename ResetFunction,
             typename Monitor = TNL::Solvers::IterativeSolverMonitor< double, int > >
   std::pair< double, double >
   timeFunction( ComputeFunction compute,
                 ResetFunction reset,
                 int maxLoops,
                 const double& minTime,
                 Monitor && monitor = Monitor() )
   {
      // the timer is constructed zero-initialized and stopped
      Timer timer;

      // set timer to the monitor
      monitor.setTimer( timer );

      // warm up
      reset();
      compute();

      Containers::Vector< double > results( maxLoops );
      results.setValue( 0.0 );

      for( loops = 0;
           loops < maxLoops || sum( results ) < minTime;
           loops++ )
      {
         // abuse the monitor's "time" for loops
         monitor.setTime( loops + 1 );
         reset();

         // Explicit synchronization of the CUDA device
#ifdef HAVE_CUDA
         if( std::is_same< Device, Devices::Cuda >::value )
            cudaDeviceSynchronize();
#endif

         // reset timer before each computation
         timer.reset();
         timer.start();
         compute();
#ifdef HAVE_CUDA
         if( std::is_same< Device, Devices::Cuda >::value )
            cudaDeviceSynchronize();
#endif
         timer.stop();

         results[ loops ] = timer.getRealTime();
      }

      const double mean = sum( results ) / (double) loops;
      if( loops > 1 ) {
         const double stddev = 1.0 / std::sqrt( loops - 1 ) * l2Norm( results - mean );
         return std::make_pair( mean, stddev );
      }
      else {
         const double stddev = std::numeric_limits<double>::quiet_NaN();
         return std::make_pair( mean, stddev );
      }
   }

   // returns a pair of (mean, stddev) where mean is the arithmetic mean of the
   // computation times and stddev is the sample standard deviation
   template< typename ComputeFunction,
             typename Monitor = TNL::Solvers::IterativeSolverMonitor< double, int > >
   std::pair< double, double >
   timeFunction( ComputeFunction compute,
                 int maxLoops,
                 const double& minTime,
                 Monitor && monitor = Monitor() )
   {
      auto noReset = [] () {};
      return timeFunction( compute, noReset, maxLoops, minTime, monitor );
   }

   int getPerformedLoops() const
   {
      return this->loops;
   }

protected:
   int loops;
};

} // namespace Benchmarks
} // namespace TNL
