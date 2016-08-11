/***************************************************************************
                          IterativeSolverMonitor_impl.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iomanip>

namespace TNL {
namespace Solvers {   

template< typename Real, typename Index>
IterativeSolverMonitor< Real, Index > :: IterativeSolverMonitor()
: iterations( 0 ),
  residue( 0 ),
  refreshing( 0 ),
  refreshRate( 1 ),
  verbose( 1 )
{
}

template< typename Real, typename Index>
void IterativeSolverMonitor< Real, Index > :: setIterations( const Index& iterations )
{
   this->iterations = iterations;
}

template< typename Real, typename Index>
const Index& IterativeSolverMonitor< Real, Index > :: getIterations() const
{
   return this->iterations;
}

template< typename Real, typename Index>
void IterativeSolverMonitor< Real, Index > :: setResidue( const Real& residue )
{
   this->residue = residue;
}

template< typename Real, typename Index>
const Real& IterativeSolverMonitor< Real, Index > :: getResidue() const
{
   return this->residue;
}

template< typename Real, typename Index>
void IterativeSolverMonitor< Real, Index > :: setVerbose( const Index& verbose )
{
   this->verbose = verbose;
}

template< typename Real, typename Index>
void IterativeSolverMonitor< Real, Index > :: setRefreshRate( const Index& refreshRate )
{
   this->refreshRate = refreshRate;
}

template< typename Real, typename Index>
void IterativeSolverMonitor< Real, Index > :: refresh( bool force )
{
   if( this->verbose > 0 && ( force || this->getIterations() % this->refreshRate == 0 ) )
   {
     std::cout << " ITER:" << std::setw( 8 ) << this->getIterations()
           << " RES:" << std::setprecision( 5 ) << std::setw( 12 ) << this->getResidue()
           << " CPU: " << std::setw( 8 ) << this->getCPUTime()
           << " ELA: " << std::setw( 8 ) << this->getRealTime()
           << "   \r" << std::flush;
   }
   this->refreshing ++;
}

template< typename Real, typename Index>
void IterativeSolverMonitor< Real, Index > :: resetTimers()
{
   cpuTimer.reset();
   rtTimer.reset();
}

template< typename Real, typename Index>
double IterativeSolverMonitor< Real, Index > :: getCPUTime()
{
   return cpuTimer.getTime();
}

template< typename Real, typename Index>
double IterativeSolverMonitor< Real, Index > :: getRealTime()
{
   return rtTimer.getTime();
}

} // namespace Solvers
} // namespace TNL
