/***************************************************************************
                          mTimerRT.h  -  description
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

#ifndef mTimerRTH
#define mTimerRTH

#include "mpi-supp.h"

class mTimerRT
{
   public:

   mTimerRT();

   void Reset();

   double GetTime() const;

   protected:

   double initial_time;
};

extern mTimerRT default_mcore_rt_timer;
#endif
