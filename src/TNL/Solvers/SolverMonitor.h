/***************************************************************************
                          SolverMonitor.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <thread>
#include <atomic>

#include <TNL/Timer.h>

namespace TNL {
namespace Solvers {   

class SolverMonitor
{
public:
   SolverMonitor()
      : timeout_milliseconds( 500 ),
        started( false ),
        stopped( false ),
        timer( nullptr )
   {}

   virtual void refresh( bool force = false ) = 0;

   void setRefreshRate( const int& refreshRate )
   {
      timeout_milliseconds = refreshRate;
   }

   void setTimer( Timer& timer )
   {
      this->timer = &timer;
   }

   void runMainLoop()
   {
      // We need to use both 'started' and 'stopped' to avoid a deadlock
      // when the loop thread runs this method delayed after the
      // SolverMonitorThread's destructor has already called stopMainLoop()
      // from the main thread.
      started = true;

      const int timeout_base = 100;
      const std::chrono::milliseconds timeout( timeout_base );

      while( ! stopped ) {
         refresh( true );

         // make sure to detect changes to refresh rate
         int steps = timeout_milliseconds / timeout_base;
         if( steps <= 0 )
            steps = 1;

         int i = 0;
         while( ! stopped && i++ < steps ) {
            std::this_thread::sleep_for( timeout );
         }
      }

      // reset to initial state
      started = false;
      stopped = false;
   }

   void stopMainLoop()
   {
      stopped = true;
   }

   bool isStopped() const
   {
      return stopped;
   }

protected:
   double getElapsedTime()
   {
      if( ! timer )
         return 0.0;
      return timer->getRealTime();
   }

   std::atomic_int timeout_milliseconds;

   std::atomic_bool started;
   std::atomic_bool stopped;

   Timer* timer;
};

// a RAII wrapper for launching the SolverMonitor's main loop in a separate thread
class SolverMonitorThread
{
   public:

   SolverMonitorThread( SolverMonitor& solverMonitor )
      : solverMonitor( solverMonitor ),
        t( &SolverMonitor::runMainLoop, &solverMonitor )
   {}

   ~SolverMonitorThread()
   {
      solverMonitor.stopMainLoop();
      if( t.joinable() )
         t.join();
   }

   private:

   SolverMonitor& solverMonitor;

   std::thread t;
};

} // namespace Solvers
} // namespace TNL
