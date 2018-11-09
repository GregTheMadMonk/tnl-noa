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

/// \brief Class for time measuring
class Timer
{
   public:

      /////
      /// \brief Basic constructor.
      ///
      /// This function creates a new timer.
      Timer();

      /////
      /// \brief Resets timer.
      ///
      /// Resets all time and cycle measurements such as real time, CPU time and CPU cycles.
      /// Sets all of them to zero.
      void reset();

      ////
      /// \brief Stops timer.
      ///
      /// Stops all time and cycle measurements such as real time, CPU time and CPU cycles.
      void stop();

      /////
      /// \brief Starts timer.
      ///
      /// Starts all time and cycle measurements such as real time, CPU time and CPU cycles.
      void start();

      /// \brief Counts the real (clock) time starting after the function \c start() is called.
      double getRealTime() const;

      /////
      /// \brief Measures the CPU time.
      ///
      /// CPU time is the time that measures how long it takes processor
      /// to complete all computations.
      double getCPUTime() const;

      /// Counts the number of CPU cycles (machine cycles).
      unsigned long long int getCPUCycles() const;

      /// \brief Writes a record to the \e logger.
      ///
      /// \param logger
      /// \param logLevel
      bool writeLog( Logger& logger, int logLevel = 0 ) const;
 
   protected:

      /// Function for reading the real time.
      double readRealTime() const;

      /// \brief Function for reading the CPU time.
      ///
      /// CPU time is the time that measures how long it takes processor
      /// to complete all computations.
      double readCPUTime() const;

      /// \brief Function for reading the number of CPU cycles (machine cycles).
      unsigned long long int readCPUCycles() const;
      

   double initialRealTime, totalRealTime,
          initialCPUTime, totalCPUTime;
 
   unsigned long long int initialCPUCycles, totalCPUCycles;

   /// \brief Saves information about state of the timer.
   ///
   /// Knows whether the timer is currently stopped or not.
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

