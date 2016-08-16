/***************************************************************************
                          TimerRT.cpp  -  description
                             -------------------
    begin                : 2007/06/26
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/TimerRT.h>
#include <TNL/tnlConfig.h>

#ifdef HAVE_SYS_TIME_H
   #include <stddef.h>
   #include <sys/time.h>
   #define HAVE_TIME
#endif

// TODO: remove this file

namespace TNL {

TimerRT defaultRTTimer;

TimerRT::TimerRT()
{
   reset();
}

void TimerRT::reset()
{
   initial_time = 0.0;
   total_time = 0.0;
   stop_state = true;
}

void TimerRT::stop()
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

void TimerRT::start()
{
#ifdef HAVE_TIME
   struct timeval tp;
   int rtn = gettimeofday( &tp, NULL );
   initial_time = ( double ) tp. tv_sec + 1.0e-6 * ( double ) tp. tv_usec;
   stop_state = false;
#endif
}

double TimerRT::getTime()
{
#ifdef HAVE_TIME
   if( ! stop_state ) {
      stop();
      start();
   }
   return total_time;
#endif
   return -1;
}

} // namespace TNL