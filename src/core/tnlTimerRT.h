/***************************************************************************
                          tnlTimerRT.h  -  description
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

#ifndef tnlTimerRTH
#define tnlTimerRTH

#include "mpi-supp.h"

class tnlTimerRT
{
   public:

   tnlTimerRT();

   void Reset();

   void Stop();

   void Continue();

   double GetTime();

   protected:

   double initial_time;

   double total_time;

   bool stop_state;
};

extern tnlTimerRT default_mcore_rt_timer;
#endif
