/***************************************************************************
                          navierStokesSolverMonitor.h  -  description
                             -------------------
    begin                : Mar 13, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef NAVIERSTOKESSOLVERMONITOR_H_
#define NAVIERSTOKESSOLVERMONITOR_H_

#include <TNL/solvers/ode/tnlODESolverMonitor.h>

template< typename Real, typename Index >
class navierStokesSolverMonitor : public tnlODESolverMonitor< Real, Index >
{
   public:

   navierStokesSolverMonitor();

   void refresh();

   Real uMax, uAvg, rhoMax, rhoAvg, rhoUMax, rhoUAvg, eMax, eAvg;

};

#include "navierStokesSolverMonitor_impl.h"

#endif /* NAVIERSTOKESSOLVERMONITOR_H_ */
