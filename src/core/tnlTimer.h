/***************************************************************************
                          tnlTimer.h  -  description
                             -------------------
    begin                : Mar 14, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
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


#ifndef TNLTIMER_H
#define	TNLTIMER_H

#include <core/tnlLogger.h>

class tnlTimer
{
   public:
   
      tnlTimer();

      void reset();

      void stop();

      void start();

      double getRealTime();

      double getCPUTime();

      unsigned long long int getCPUCycles();
      
      bool writeLog( tnlLogger& logger, int logLevel = 0 );
         
   protected:

   double initialRealTime, totalRealTime,
          initialCPUTime, totalCPUTime;
   
   unsigned long long int initialCPUCycles, totalCPUCycles;
   
   bool stopState;
   
   inline unsigned long long rdtsc()
   {
     unsigned hi, lo;
     __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
     return ( ( unsigned long long ) lo ) | ( ( ( unsigned long long ) hi ) << 32 );
   }
};

extern tnlTimer defaultTimer;

#endif	/* TNLTIMER_H */

