/***************************************************************************
                          TimerCPU.h  -  description
                             -------------------
    begin                : 2007/06/23
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef tnlTimerCPUH
#define tnlTimerCPUH

// TODO: remove this file

#include <TNL/tnlConfig.h>
#ifdef HAVE_SYS_RESOURCE_H
   #include <sys/resource.h>
#endif

#include <TNL/core/mpi-supp.h>

namespace TNL {

class TimerCPU
{
   public:

   TimerCPU();

   void reset();
 
   void stop();

   void start();

   double getTime( int root = 0, MPI_Comm = MPI_COMM_WORLD );
 
   protected:

   double initial_time;

   double total_time;

   bool stop_state;
};

extern TimerCPU defaultCPUTimer;

} // namespace TNL

#endif
