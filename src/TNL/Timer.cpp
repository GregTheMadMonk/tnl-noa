/***************************************************************************
                          Timer.cpp  -  description
                             -------------------
    begin                : Mar 14, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Timer.h>
#include <TNL/Logger.h>

#include <TNL/tnlConfig.h>
#ifdef HAVE_SYS_RESOURCE_H
   #include <sys/resource.h>
#endif
#ifdef HAVE_SYS_TIME_H
   #include <stddef.h>
   #include <sys/time.h>
   #define HAVE_TIME
#endif

namespace TNL {

Timer defaultTimer;

Timer::Timer()
{
   reset();
}

void Timer::reset()
{
   this->initialCPUTime = 0;
   this->totalCPUTime = 0.0;
   this->initialRealTime = 0;
   this->totalRealTime = 0.0;
   this->initialCPUCycles = 0;
   this->totalCPUCycles = 0;
   this->stopState = true;
}

void Timer::stop()
{

   if( ! this->stopState )
   {
      this->totalRealTime += this->readRealTime() - this->initialRealTime;
      this->totalCPUTime += this->readCPUTime() - this->initialCPUTime;
      this->totalCPUCycles += this->readCPUCycles() - this->initialCPUCycles;
      this->stopState = true;
   }
}

void Timer::start()
{
   this->initialRealTime = this->readRealTime();
   this->initialCPUTime = this->readCPUTime();
   this->initialCPUCycles = this->readCPUCycles(); 
   this->stopState = false;
}

double Timer::getRealTime() const
{
   if( ! this->stopState )
    return this->readRealTime() - this->initialRealTime;
   return this->totalRealTime;
}

double Timer::getCPUTime() const
{
   if( ! this->stopState )
    return this->readCPUTime() - this->initialCPUTime;
   return this->totalCPUTime;
}

unsigned long long int Timer::getCPUCycles() const
{
   if( ! this->stopState )
    return this->readCPUCycles() - this->initialCPUCycles;
   return this->totalCPUCycles;
}

double Timer::readRealTime() const
{
#ifdef HAVE_TIME
   struct timeval tp;
   int rtn = gettimeofday( &tp, NULL );
   return ( double ) tp. tv_sec + 1.0e-6 * ( double ) tp. tv_usec;
#else
   return -1;
#endif
}

double Timer::readCPUTime() const
{
#ifdef HAVE_SYS_RESOURCE_H
   rusage initUsage;
   getrusage( RUSAGE_SELF, &initUsage );
   return initUsage. ru_utime. tv_sec + 1.0e-6 * ( double ) initUsage. ru_utime. tv_usec;
#else
   return -1;
#endif
}

unsigned long long int Timer::readCPUCycles() const
{
   return this->rdtsc();
}


bool Timer::writeLog( Logger& logger, int logLevel ) const
{
   logger.writeParameter< double                 >( "Real time:",  this->getRealTime(),  logLevel );
   logger.writeParameter< double                 >( "CPU time:",   this->getCPUTime(),   logLevel );
   logger.writeParameter< unsigned long long int >( "CPU Cycles:", this->getCPUCycles(), logLevel );
   return true;
}

} // namespace TNL
