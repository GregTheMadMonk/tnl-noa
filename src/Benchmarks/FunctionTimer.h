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
#include <TNL/Solvers/IterativeSolverMonitor.h>

namespace TNL {
   namespace Benchmarks {


template< typename Device,
          bool timing >
class FunctionTimer
{
   public:
      using DeviceType = Device;

      template< typename ComputeFunction,
                typename ResetFunction,
                typename Monitor = TNL::Solvers::IterativeSolverMonitor< double, int > >
      static double
      timeFunction( ComputeFunction compute,
                    ResetFunction reset,
                    int maxLoops,
                    const double& minTime,
                    int verbose = 1,
                    Monitor && monitor = Monitor(),
                    bool performReset = true )
      {
         // the timer is constructed zero-initialized and stopped
         Timer timer;

         // set timer to the monitor
         if( verbose > 1 )
            monitor.setTimer( timer );

         // warm up
         reset();
         compute();

         int loops;
         // If we do not perform reset function and don't need
         // the monitor, the timer is not interrupted after each loop.
         if( ! performReset && verbose < 2 )
         {
            if( timing )
               timer.start();
            // Explicit synchronization of the CUDA device
#ifdef HAVE_CUDA      
               if( std::is_same< Device, Devices::Cuda >::value )
                  cudaDeviceSynchronize();
#endif            
            for( loops = 0;
                 loops < maxLoops || ( timing && timer.getRealTime() < minTime );
                 ++loops) 
               compute();
            // Explicit synchronization of the CUDA device
#ifdef HAVE_CUDA
            if( std::is_same< Device, Devices::Cuda >::value )
               cudaDeviceSynchronize();
#endif
            if( timing )
               timer.stop();
         }
         else
         {
            for( loops = 0;
                 loops < maxLoops || ( timing && timer.getRealTime() < minTime );
                 ++loops) 
            {
               // abuse the monitor's "time" for loops
               monitor.setTime( loops + 1 );

               reset();

               // Explicit synchronization of the CUDA device
#ifdef HAVE_CUDA      
               if( std::is_same< Device, Devices::Cuda >::value )
                  cudaDeviceSynchronize();
#endif
               if( timing )
                  timer.start();
               compute();
#ifdef HAVE_CUDA
               if( std::is_same< Device, Devices::Cuda >::value )
                  cudaDeviceSynchronize();
#endif
               if( timing )
                  timer.stop();
            }
         }
         if( timing )
            return timer.getRealTime() / ( double ) loops;
         else
            return std::numeric_limits<double>::quiet_NaN();
      }

      template< typename ComputeFunction,
                typename Monitor = TNL::Solvers::IterativeSolverMonitor< double, int > >
      static double
      timeFunction( ComputeFunction compute,
                    int maxLoops,
                    const double& minTime,
                    int verbose = 1,
                    Monitor && monitor = Monitor() )
      {
         auto noReset = [] () {};
         return timeFunction( compute, noReset, maxLoops, minTime, verbose, monitor, false );
      }
};

   } // namespace Benchmarks
} // namespace TNL
