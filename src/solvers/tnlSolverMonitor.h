/***************************************************************************
                          tnlSolverMonitor.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
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

#ifndef TNLSOLVERMONITOR_H_
#define TNLSOLVERMONITOR_H_

template< typename Real, typename Index >
class tnlSolverMonitor
{
   public:

   virtual void refresh() = 0;

   ~tnlSolverMonitor() {};
      
};


#endif /* TNLSOLVERMONITOR_H_ */
