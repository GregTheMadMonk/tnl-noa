/***************************************************************************
                          tnlIterativeSolverMonitor.h  -  description
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

#ifndef TNLITERATIVESOLVERMONITOR_H_
#define TNLITERATIVESOLVERMONITOR_H_

#include <solvers/tnlSolverMonitor.h>
#include <solvers/tnlIterativeSolver.h>
#include <core/tnlTimerCPU.h>
#include <core/tnlTimerRT.h>

template< typename Real, typename Index>
class tnlIterativeSolverMonitor : public tnlSolverMonitor< Real, Index >
{
   public:

   typedef Index IndexType;
   typedef Real RealType;

   tnlIterativeSolverMonitor();

   void setIterations( const IndexType& iterations );

   const IndexType& getIterations() const;

   void setResidue( const RealType& residue );

   const RealType& getResidue() const;

   void setVerbose( const Index& verbose );

   virtual void refresh();

   void resetTimers();

   double getCPUTime();

   double getRealTime();

   protected:

   IndexType iterations;

   RealType residue;

   IndexType refreshing;

   IndexType outputPeriod;

   IndexType verbose;

   tnlTimerCPU cpuTimer;

   tnlTimerRT rtTimer;
};

#include <implementation/solvers/tnlIterativeSolverMonitor_impl.h>

#endif /* TNLITERATIVESOLVERMONITOR_H_ */
