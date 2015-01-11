/***************************************************************************
                          tnlTimerCPU.cpp  -  description
                             -------------------
    begin                : 2007/06/23
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include <core/tnlTimerCPU.h>

tnlTimerCPU defaultCPUTimer;

tnlTimerCPU :: tnlTimerCPU()
{
   Reset();
}
//--------------------------------------------------------------------------
void tnlTimerCPU :: Reset()
{
#ifdef HAVE_SYS_RESOURCE_H
   rusage init_usage;
   getrusage(  RUSAGE_SELF, &init_usage );
   initial_time = init_usage. ru_utime. tv_sec + 1.0e-6 * ( double ) init_usage. ru_utime. tv_usec;
#else
   initial_time = 0;
#endif
   total_time = 0.0;
   stop_state = false;
}
//--------------------------------------------------------------------------
void tnlTimerCPU :: Stop()
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
//--------------------------------------------------------------------------
void tnlTimerCPU :: Continue()
{
#ifdef HAVE_SYS_RESOURCE_H
   rusage init_usage;
   getrusage(  RUSAGE_SELF, &init_usage );
   initial_time = init_usage. ru_utime. tv_sec + 1.0e-6 * ( double ) init_usage. ru_utime. tv_usec;
#endif
  stop_state = false;
}
//--------------------------------------------------------------------------
double tnlTimerCPU :: GetTime( int root, MPI_Comm comm )
{
#ifdef HAVE_SYS_RESOURCE_H
   Stop();
   Continue();
   double mpi_total_time;
   MPIReduce( total_time, mpi_total_time, 1, MPI_SUM, root, comm );
   return mpi_total_time;
#else
   return -1;
#endif
}
