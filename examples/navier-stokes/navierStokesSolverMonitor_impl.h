/***************************************************************************
                          navierStokesSolverMonitor_impl.h  -  description
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

#ifndef TNLNAVIERSTOKESSOLVERMONITOR_IMPL_H_
#define TNLNAVIERSTOKESSOLVERMONITOR_IMPL_H_

#include <fstream>

using namespace std;

template< typename Real, typename Index >
navierStokesSolverMonitor< Real, Index > :: navierStokesSolverMonitor()
{
}

template< typename Real, typename Index >
void navierStokesSolverMonitor< Real, Index > :: refresh()
{
   if( this->verbose > 0 && this->refresRate % this->refreshRate == 0 )
   {
      cout << "V=( " << uMax
           << " , " << uAvg
           << " ) E=( " << eMax
           << ", " << eAvg << " ) ";
   }
   tnlODESolverMonitor< Real, Index > :: refresh();
}

#endif /* TNLNAVIERSTOKESSOLVERMONITOR_IMPL_H_ */
