/***************************************************************************
                          tnlTimerRT.cpp  -  description
                             -------------------
    begin                : 2007/06/26
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

#include <core/tnlTimerRT.h>
#include <tnlConfig.h>

#ifdef HAVE_SYS_TIME_H
   #include <stddef.h>
   #include <sys/time.h>
   #define HAVE_TIME
#endif


tnlTimerRT defaultRTTimer;

tnlTimerRT::tnlTimerRT()
{
   reset();
}

void tnlTimerRT::reset()
{
   initial_time = 0.0;
   total_time = 0.0;
   stop_state = true;
}

void tnlTimerRT::stop()
{
#ifdef HAVE_TIME
   if( ! stop_state )
   {
      struct timeval tp;
      int rtn = gettimeofday( &tp, NULL );
      total_time += ( double ) tp. tv_sec + 1.0e-6 * ( double ) tp. tv_usec - initial_time;
      stop_state = true;
   }
#endif
}

void tnlTimerRT::start()
{
#ifdef HAVE_TIME
   struct timeval tp;
   int rtn = gettimeofday( &tp, NULL );
   initial_time = ( double ) tp. tv_sec + 1.0e-6 * ( double ) tp. tv_usec;
   stop_state = false;
#endif
}

double tnlTimerRT::getTime()
{
#ifdef HAVE_TIME
   if( ! stop_state )
   {
      stop();
      start();
   }
   return total_time;
#endif
   return -1;
}
