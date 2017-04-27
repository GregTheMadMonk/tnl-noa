/***************************************************************************
                          SolverMonitor.cpp  -  description
                             -------------------
    begin                : Feb 10, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Solvers/SolverMonitor.h>

namespace TNL {
namespace Solvers {   

   SolverMonitor::SolverMonitor()
      : timeout_milliseconds(500),
        stopped(true),
        timer(nullptr)
   {};

   SolverMonitor::~SolverMonitor() {};
 

   void SolverMonitor::setRefreshRate( const int& refreshRate )
   {
      timeout_milliseconds = refreshRate;
   }

   void SolverMonitor::setTimer( Timer& timer )
   {
      this->timer = &timer;
   }

   void SolverMonitor::runMainLoop()
   {
      stopped = false;

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
   }

   void SolverMonitor::stopMainLoop()
   {
      stopped = true;
   }

   double SolverMonitor::getElapsedTime()
   {
      if( ! timer )
         return 0.0;
      return timer->getRealTime();
   }

// a RAII wrapper for launching the SolverMonitor's main loop in a separate thread

   SolverMonitorThread::SolverMonitorThread( SolverMonitor& solverMonitor )
      : solverMonitor( solverMonitor ),
        t( &SolverMonitor::runMainLoop, &solverMonitor )
   {}

   SolverMonitorThread::~SolverMonitorThread()
   {
      solverMonitor.stopMainLoop();
      if( t.joinable() )
         t.join();
   }


} // namespace Solvers
} // namespace TNL

