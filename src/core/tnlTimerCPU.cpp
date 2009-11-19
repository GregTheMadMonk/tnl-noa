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

#include "config.h"
#ifdef HAVE_SYS_RESOURCE_H
   #include <sys/resource.h>
#endif
#include "tnlTimerCPU.h"

tnlTimerCPU default_mcore_cpu_timer;

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
   initial_time = init_usage. ru_utime. tv_sec;
#else
   initial_time = 0;
#endif
}
//--------------------------------------------------------------------------  
int tnlTimerCPU :: GetTime( int root, MPI_Comm comm ) const
{
#ifdef HAVE_SYS_RESOURCE_H
   rusage cur_usage;
   getrusage( RUSAGE_SELF, &cur_usage );
   int time = cur_usage. ru_utime. tv_sec - initial_time;
   int total_time;
   MPIReduce( time, total_time, 1, MPI_SUM, root, comm ); 
   return total_time;
#else
   return -1;
#endif
}
