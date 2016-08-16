/***************************************************************************
                          IterativeSolver.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Solvers/IterativeSolverMonitor.h>

namespace TNL {
namespace Solvers {   

template< typename Real, typename Index >
class IterativeSolver
{
   public:

   IterativeSolver();

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

   void setMaxIterations( const Index& maxIterations );

   const Index& getMaxIterations() const;

   void setMinIterations( const Index& minIterations );

   const Index& getMinIterations() const;

   const Index& getIterations() const;

   void setConvergenceResidue( const Real& convergenceResidue );

   const Real& getConvergenceResidue() const;

   void setDivergenceResidue( const Real& divergenceResidue );

   const Real& getDivergenceResidue() const;

   void setResidue( const Real& residue );

   const Real& getResidue() const;

   void setRefreshRate( const Index& refreshRate );

   void setSolverMonitor( IterativeSolverMonitor< Real, Index >& solverMonitor );

   void resetIterations();

   bool nextIteration();

   bool checkNextIteration();

   bool checkConvergence();

   void refreshSolverMonitor( bool force = false );


   protected:

   Index maxIterations;

   Index minIterations;

   Index currentIteration;

   Real convergenceResidue;

   /****
    * If the current residue is over divergenceResidue the solver is stopped.
    */
   Real divergenceResidue;

   Real currentResidue;

   IterativeSolverMonitor< Real, Index >* solverMonitor;

   Index refreshRate;
};

} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/IterativeSolver_impl.h>