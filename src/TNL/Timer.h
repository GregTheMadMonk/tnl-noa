/***************************************************************************
                          Timer.h  -  description
                             -------------------
    begin                : Mar 14, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#pragma once

namespace TNL {

class Logger;

class Timer
{
   public:
 
      Timer();

      void reset();

      void stop();

      void start();

      double getRealTime() const;

      double getCPUTime() const;

      unsigned long long int getCPUCycles() const;
 
      bool writeLog( Logger& logger, int logLevel = 0 ) const;
 
   protected:
      
      double readRealTime() const;

      double readCPUTime() const;

      unsigned long long int readCPUCycles() const;
      

   double initialRealTime, totalRealTime,
          initialCPUTime, totalCPUTime;
 
   unsigned long long int initialCPUCycles, totalCPUCycles;
 
   bool stopState;
 
   inline unsigned long long rdtsc() const
   {
     unsigned hi, lo;
     __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
     return ( ( unsigned long long ) lo ) | ( ( ( unsigned long long ) hi ) << 32 );
   }
};

extern Timer defaultTimer;

} // namespace TNL

