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

/// \brief Class for time measuring.
///
/// Counts the elapsed time in seconds between the start() and stop() methods.
/// \par Example
/// \include TimerExample.cpp
// \par Output
// \include TimerExample.out
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
      /// CPU cycles. Method start() can be used also after using stop() method.
      /// The timer then continues measuring the time without reseting.
      void start();

      /////
      /// \brief Returns the elapsed time on given timer.
      ///
      /// It returns the elapsed time (in seconds) between calling the start() and stop() methods.
      /// Starts counting the real time after the method start() is called and
      /// pauses when the method stop() is called.
      /// If the timer has been started more then once without resetting,
      /// the real time is counted by adding all intervals (between start and stop
      /// methods) together.
      /// This function can be called while the timer is running, there is no
      /// need to use stop() method first.
      double getRealTime() const;

      /////
      /// \brief Returns the elapsed CPU time on given timer.
      ///
      /// The CPU time is measured in seconds.
      /// CPU time is the amount of time for which a central processing unit (CPU)
      /// was used for processing instructions of a computer program or operating system.
      /// The CPU time is measured by adding the amount of CPU time between start() and stop()
      /// methods together.
      double getCPUTime() const;

      /// \brief Returns the number of CPU cycles (machine cycles).
      ///
      /// CPU cycles are counted by adding the number of CPU cycles between start() and stop()
      /// methods together.
      unsigned long long int getCPUCycles() const;

      /// \brief Writes a record into the \e logger.
      ///
      /// \param logger
      /// \param logLevel A non-negative integer recording the log record indent.
      bool writeLog( Logger& logger, int logLevel = 0 ) const;
 
   protected:

      /// \brief Function for measuring the real time.
      ///
      /// Returns number of seconds since Epoch, 1970-01-01 00:00:00 UTC.
      double readRealTime() const;

      /// \brief Function for measuring the CPU time.
      ///
      /// CPU time is the amount of time for which a central processing unit (CPU)
      /// was used for processing instructions of a computer program or operating system.
      double readCPUTime() const;

      /// \brief Function for counting the number of CPU cycles (machine cycles).
      unsigned long long int readCPUCycles() const;
      

   double initialRealTime, totalRealTime,
          initialCPUTime, totalCPUTime;
 
   unsigned long long int initialCPUCycles, totalCPUCycles;

   /// \brief Saves information about the state of given timer.
   ///
   /// Knows whether the timer is currently stopped or it is running.
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

