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

#include "tnlConfig.h"
#ifdef HAVE_SYS_RESOURCE_H
   #include <sys/resource.h>
#endif


#include "mpi-supp.h"

class tnlTimerCPU
{
   public:

   tnlTimerCPU();

   void reset();
   
   void stop();

   void start();

   double getTime( int root = 0, MPI_Comm = MPI_COMM_WORLD );
      
   protected:

   double initial_time;

   double total_time;

   bool stop_state;
};

extern tnlTimerCPU defaultCPUTimer;



#endif
