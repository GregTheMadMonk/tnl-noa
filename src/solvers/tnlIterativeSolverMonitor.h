/***************************************************************************
                          tnlIterativeSolverMonitor.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <solvers/tnlSolverMonitor.h>
#include <core/tnlTimerCPU.h>
#include <core/tnlTimerRT.h>

namespace TNL {

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
 
   void setRefreshRate( const IndexType& refreshRate );

   virtual void refresh( bool force = false );

   void resetTimers();

   double getCPUTime();

   double getRealTime();

   protected:

   IndexType iterations;

   RealType residue;

   IndexType refreshing;

   IndexType refreshRate;

   IndexType verbose;

   tnlTimerCPU cpuTimer;

   tnlTimerRT rtTimer;
};

} // namespace TNL

#include <solvers/tnlIterativeSolverMonitor_impl.h>
