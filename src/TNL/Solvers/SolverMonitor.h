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

   SolverMonitor();

   ~SolverMonitor();
 
   virtual void refresh( bool force = false ) = 0;

   void setRefreshRate( const int& refreshRate );

   void setTimer( Timer& timer );

   void runMainLoop();

   void stopMainLoop();

   protected:

   double getElapsedTime();

   std::atomic_int timeout_milliseconds;

   std::atomic_bool stopped;

   Timer* timer;
};

// a RAII wrapper for launching the SolverMonitor's main loop in a separate thread
class SolverMonitorThread
{
   public:

   SolverMonitorThread( SolverMonitor& solverMonitor );
   ~SolverMonitorThread();
   
   private:

   SolverMonitor& solverMonitor;

   std::thread t;
};

} // namespace Solvers
} // namespace TNL

