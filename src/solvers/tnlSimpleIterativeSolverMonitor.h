/***************************************************************************
                          tnlSimpleIterativeSolverMonitor.h  -  description
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

#ifndef TNLSIMPLEITERATIVESOLVERMONITOR_H_
#define TNLSIMPLEITERATIVESOLVERMONITOR_H_

#include <solvers/tnlSolverMonitor.h>
#include <solvers/tnlIterativeSolver.h>
#include <core/tnlTimerCPU.h>
#include <core/tnlTimerRT.h>

template< typename Real, typename Index>
class tnlSimpleIterativeSolverMonitor : public tnlSolverMonitor< Real, Index >
{
   public:

   tnlSimpleIterativeSolverMonitor();

   void setSolver( const tnlIterativeSolver< Real, Index >& solver );

   void setVerbose( const Index& verbose );

   void refresh();

   void resetTimers();

   double getCPUTime();

   double getRealTime();

   protected:

   Index refreshing;

   Index outputPeriod;

   Index verbose;

   const tnlIterativeSolver< Real, Index >* solver;

   tnlTimerCPU cpuTimer;

   tnlTimerRT rtTimer;
};

template< typename Real, typename Index>
tnlSimpleIterativeSolverMonitor< Real, Index > :: tnlSimpleIterativeSolverMonitor()
: refreshing( 0 ),
  outputPeriod( 1 ),
  verbose( 1 ),
  solver( 0 )
{
}

template< typename Real, typename Index>
void tnlSimpleIterativeSolverMonitor< Real, Index > :: setSolver( const tnlIterativeSolver< Real, Index >& solver )
{
   this -> solver = &solver;
}

template< typename Real, typename Index>
void tnlSimpleIterativeSolverMonitor< Real, Index > :: setVerbose( const Index& verbose )
{
   this -> verbose = &verbose;
}

template< typename Real, typename Index>
void tnlSimpleIterativeSolverMonitor< Real, Index > :: refresh()
{
   if( ! solver )
      return;
   this -> refreshing ++;
   if( this -> verbose > 0 )
   {
      cout << " ITER:" << setw( 8 ) << solver -> getIterations()
           << " RES:" << setprecision( 5 ) << setw( 12 ) << solver -> getResidue()
           << " CPU: " << setw( 8 ) << this -> getCPUTime()
           << " ELA: " << setw( 8 ) << this -> getRealTime()
           << "   \r" << flush;
   }
}

template< typename Real, typename Index>
void tnlSimpleIterativeSolverMonitor< Real, Index > :: resetTimers()
{
   cpuTimer. Reset();
   rtTimer. Reset();
}

template< typename Real, typename Index>
double tnlSimpleIterativeSolverMonitor< Real, Index > :: getCPUTime()
{
   return cpuTimer. GetTime();
}

template< typename Real, typename Index>
double tnlSimpleIterativeSolverMonitor< Real, Index > :: getRealTime()
{
   return rtTimer. GetTime();
}


#endif /* TNLSIMPLEITERATIVESOLVERMONITOR_H_ */
