/***************************************************************************
                          tnlTimerCPU.h  -  description
                             -------------------
    begin                : 2007/06/23
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

#ifndef tnlTimerCPUH
#define tnlTimerCPUH

#include "mpi-supp.h"

class tnlTimerCPU
{
   public:

   tnlTimerCPU();

   void Reset();
   
   int GetTime( int root = 0, MPI_Comm = MPI_COMM_WORLD ) const;
      
   protected:

   int initial_time;   
};

extern tnlTimerCPU default_mcore_cpu_timer;

#endif
