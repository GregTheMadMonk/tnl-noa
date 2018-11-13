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
      /// This function creates a new timer and resets it.
      Timer();

      /////
      /// \brief Resets timer.
      ///
      /// Resets all time and cycle measurements such as real time, CPU time and CPU cycles.
      /// Sets all of them to zero.
      void reset();

      ////
      /// \brief Stops (pauses) the timer.
      ///
      /// Pauses all time and cycle measurements such as real time, CPU time and
      /// CPU cycles, but does not set them to zero.
      void stop();

      /////
      /// \brief Starts timer.
      ///
      /// Starts all time and cycle measurements such as real time, CPU time and
      /// CPU cycles. Function start() can be used also after using stop() function.
      /// The timer then continues measuring the time without reseting.
      void start();

      /// \brief Returs the real (clock/timer) time.
      ///
      /// It returns the elapsed time between calling the start() and stop() functions.
      /// Starts counting the real time after the function start() is called and
      /// pauses when the function stop() is called.
      /// If the timer have been started more then one time without resetting,
      /// the real time is counted by adding all intervals (between start and stop
      /// functions) together.
      /// This function can be called while the timer is running, there is no
      /// need to use stop() function first.
      double getRealTime() const;

      /////
      /// \brief Returns the CPU time.
      ///
      /// CPU time is the time that measures how long it takes processor
      /// to complete all computations.
      double getCPUTime() const;

      /// Returns the number of CPU cycles (machine cycles).
      unsigned long long int getCPUCycles() const;

      /// \brief Writes a record to the \e logger.
      ///
      /// \param logger
      /// \param logLevel A whole number from zero up, which indicates the indent.
      bool writeLog( Logger& logger, int logLevel = 0 ) const;
 
   protected:

      /// Function for measuring the real time.
      double readRealTime() const;

      /// \brief Function for measuring the CPU time.
      ///
      /// CPU time is the time that measures how long it takes processor
      /// to complete all computations.
      double readCPUTime() const;

      /// \brief Function for counting the number of CPU cycles (machine cycles).
      unsigned long long int readCPUCycles() const;
      

   double initialRealTime, totalRealTime,
          initialCPUTime, totalCPUTime;
 
   unsigned long long int initialCPUCycles, totalCPUCycles;

   /// \brief Saves information about state of the timer.
   ///
   /// Knows whether the timer is currently stopped or not.
   bool stopState;

   /// \brief Time Stamp Counter returning number of CPU cycles since reset.
   ///
   /// Only for x86 compatibile CPUs.
   inline unsigned long long rdtsc() const
   {
     unsigned hi, lo;
     __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
     return ( ( unsigned long long ) lo ) | ( ( ( unsigned long long ) hi ) << 32 );
   }
};

// !!! Odstranit ???!!!
extern Timer defaultTimer;

} // namespace TNL

