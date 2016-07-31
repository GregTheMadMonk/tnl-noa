/***************************************************************************
                          TimerCPU.cpp  -  description
                             -------------------
    begin                : 2007/06/23
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/TimerCPU.h>

// TODO: remove this file

namespace TNL {

TimerCPU defaultCPUTimer;

TimerCPU :: TimerCPU()
{
   reset();
}

void TimerCPU::reset()
{
   initial_time = 0;
   total_time = 0.0;
   stop_state = true;
}

void TimerCPU::stop()
{
#ifdef HAVE_SYS_RESOURCE_H
   if( ! stop_state )
   {
      rusage init_usage;
      getrusage(  RUSAGE_SELF, &init_usage );
      total_time += init_usage. ru_utime. tv_sec + 1.0e-6 * ( double ) init_usage. ru_utime. tv_usec - initial_time;
      stop_state = true;
   }
#endif
}

void TimerCPU::start()
{
#ifdef HAVE_SYS_RESOURCE_H
   rusage init_usage;
   getrusage(  RUSAGE_SELF, &init_usage );
   initial_time = init_usage. ru_utime. tv_sec + 1.0e-6 * ( double ) init_usage. ru_utime. tv_usec;
#endif
   stop_state = false;
}

double TimerCPU::getTime( int root, MPI_Comm comm )
{
#ifdef HAVE_SYS_RESOURCE_H
   if( ! stop_state ) {
      stop();
      start();
   }
   double mpi_total_time;
   MPIReduce( total_time, mpi_total_time, 1, MPI_SUM, root, comm );
   return mpi_total_time;
#else
   return -1;
#endif
}


} // namespace TNL