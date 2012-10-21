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

template< typename Real, typename Index>
class tnlSimpleIterativeSolverMonitor : public tnlSolverMonitor< Real, Index >
{
   public:

   tnlSimpleIterativeSolverMonitor();

   void setSolver( const tnlIterativeSolver< Real, Index >& solver );

   void setVerbose( const Index& verbose );

   void refresh();

   protected:

   Index refreshing;

   Index outputPeriod;

   Index verbose;

   const tnlIterativeSolver< Real, Index >* solver;
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
           << " RES:" << setprecision( 5 ) << setw( 12 ) << solver -> getResidue();
/*      if( this -> cpu_timer )
         cout << " CPU: " << setw( 8 ) << cpu_time;
      if( this -> rt_timer )
         cout << " ELA: " << setw( 8 ) << this -> rt_timer -> GetTime();*/
      cout << "   \r" << flush;
   }
}

#endif /* TNLSIMPLEITERATIVESOLVERMONITOR_H_ */
