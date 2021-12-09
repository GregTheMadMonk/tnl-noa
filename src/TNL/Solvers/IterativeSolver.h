/***************************************************************************
                          IterativeSolver.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <limits>

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Solvers/IterativeSolverMonitor.h>

namespace TNL {
namespace Solvers {

template< typename Real,
          typename Index,
          typename SolverMonitor = IterativeSolverMonitor< Real, Index > >
class IterativeSolver
{
public:
   using SolverMonitorType = SolverMonitor;

   IterativeSolver() = default;

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

   void setSolverMonitor( SolverMonitorType& solverMonitor );

   void resetIterations();

   bool nextIteration();

   bool checkNextIteration();

   bool checkConvergence();

   void refreshSolverMonitor( bool force = false );

protected:
   Index maxIterations = 1000000000;

   Index minIterations = 0;

   Index currentIteration = 0;

   Real convergenceResidue = 1e-6;

   // If the current residue is greater than divergenceResidue, the solver is stopped.
   Real divergenceResidue = std::numeric_limits< float >::max();

   Real currentResidue = 0;

   SolverMonitor* solverMonitor = nullptr;

   Index refreshRate = 1;

   String residualHistoryFileName = "";

   std::ofstream residualHistoryFile;
};

} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/IterativeSolver.hpp>
