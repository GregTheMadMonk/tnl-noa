/***************************************************************************
                          TimerRT.h  -  description
                             -------------------
    begin                : 2007/06/26
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef tnlTimerRTH
#define tnlTimerRTH

#include <TNL/core/mpi-supp.h>

// TODO: remove this file

namespace TNL {

class TimerRT
{
   public:

   TimerRT();

   void reset();

   void stop();

   void start();

   double getTime();

   protected:

   double initial_time;

   double total_time;

   bool stop_state;
};

extern TimerRT defaultRTTimer;

} // namespace TNL

#endif
