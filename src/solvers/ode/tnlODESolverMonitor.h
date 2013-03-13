/***************************************************************************
                          tnlODESolverMonitor.h  -  description
                             -------------------
    begin                : Mar 12, 2013
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

#ifndef TNLODESOLVERMONITOR_H_
#define TNLODESOLVERMONITOR_H_

#include <solvers/tnlIterativeSolverMonitor.h>

template< typename Real, typename Index>
class tnlODESolverMonitor : public tnlIterativeSolverMonitor< Real, Index >
{
   public:

   typedef Real RealType;
   typedef Index IndexType;

   tnlODESolverMonitor();

   void refresh();

   void setTimeStep( const RealType& timeStep );

   const RealType& getTimeStep() const;

   void setTime( const RealType& time );

   const RealType& getTime() const;

   protected:

   RealType timeStep;

   RealType time;

};

#include <implementation/solvers/ode/tnlODESolverMonitor_impl.h>

#endif /* TNLODESOLVERMONITOR_H_ */
