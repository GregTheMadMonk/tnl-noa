/***************************************************************************
                          tnlTimerRT.h  -  description
                             -------------------
    begin                : 2007/06/26
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef tnlTimerRTH
#define tnlTimerRTH

#include "mpi-supp.h"

// TODO: remove this file

namespace TNL {

class tnlTimerRT
{
   public:

   tnlTimerRT();

   void reset();

   void stop();

   void start();

   double getTime();

   protected:

   double initial_time;

   double total_time;

   bool stop_state;
};

extern tnlTimerRT defaultRTTimer;

} // namespace TNL

#endif
