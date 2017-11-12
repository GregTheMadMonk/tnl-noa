/***************************************************************************
                          IterativeSolverMonitor.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Solvers/SolverMonitor.h>

namespace TNL {
namespace Solvers {   

template< typename Real, typename Index>
class IterativeSolverMonitor : public SolverMonitor
{
public:
   typedef Index IndexType;
   typedef Real RealType;

   IterativeSolverMonitor();

   void setStage( const std::string& stage );

   void setTime( const RealType& time );

   void setTimeStep( const RealType& timeStep );

   void setIterations( const IndexType& iterations );

   void setResidue( const RealType& residue );

   void setVerbose( const Index& verbose );
 
   virtual void refresh();

protected:
   int getLineWidth();

   std::string stage, saved_stage;

   std::atomic_bool saved;

   RealType time, saved_time, timeStep, saved_timeStep, residue, saved_residue;

   IndexType iterations, saved_iterations;

   IndexType verbose;
};

} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/IterativeSolverMonitor_impl.h>
