/***************************************************************************
                          mTimerRT.cpp  -  description
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

#include "mTimerRT.h"
#include "config.h"
#ifdef HAVE_STDDEF_H
   #ifdef HAVE_SYS_TIME_H 
      #include <stddef.h>
      #include <sys/time.h>
      #define HAVE_TIME
   #endif
#endif

mTimerRT default_mcore_rt_timer;
//--------------------------------------------------------------------------
mTimerRT :: mTimerRT()
{
   Reset();
}
//--------------------------------------------------------------------------
void mTimerRT :: Reset()
{
#ifdef HAVE_TIME
   struct timeval tp;
   int rtn = gettimeofday( &tp, NULL );
   initial_time = ( double ) tp. tv_sec + 1.0e-6 * ( double ) tp. tv_usec;   
   return;
#endif
   initial_time = 0.0;
}
//--------------------------------------------------------------------------
double mTimerRT :: GetTime() const
{
#ifdef HAVE_TIME
   struct timeval tp;
   int rtn = gettimeofday( &tp, NULL );
   return ( double ) tp. tv_sec + 1.0e-6 * ( double ) tp. tv_usec - initial_time;   
#endif
 return -1;
}
