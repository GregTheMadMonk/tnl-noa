/***************************************************************************
                          IterativeSolver_impl.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <cmath>

#include "IterativeSolver.h"

namespace TNL {
namespace Solvers {

template< typename Real, typename Index, typename SolverMonitor >
void
IterativeSolver< Real, Index, SolverMonitor >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   config.addEntry< int >   ( prefix + "max-iterations", "Maximal number of iterations the solver may perform.", 1000000000 );
   config.addEntry< int >   ( prefix + "min-iterations", "Minimal number of iterations the solver must perform.", 0 );

   // The default value for the convergence residue MUST be zero since not in all problems we want to stop the solver
   // when we reach a state near a steady state. This can be only temporary if, for example, when the boundary conditions
   // are time dependent (growing velocity at inlet starting from 0).
   config.addEntry< double >( prefix + "convergence-residue", "Convergence occurs when the residue drops bellow this limit.", 0.0 );
   config.addEntry< double >( prefix + "divergence-residue", "Divergence occurs when the residue exceeds given limit.", std::numeric_limits< float >::max() );
   // TODO: setting refresh rate should be done in SolverStarter::setup (it's not a parameter of the IterativeSolver)
   config.addEntry< int >   ( prefix + "refresh-rate", "Number of milliseconds between solver monitor refreshes.", 500 );

   config.addEntry< String >( prefix + "residual-history-file", "Path to the file where the residual history will be saved.", "" );
}

template< typename Real, typename Index, typename SolverMonitor >
bool
IterativeSolver< Real, Index, SolverMonitor >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   this->setMaxIterations( parameters.getParameter< int >( prefix + "max-iterations" ) );
   this->setMinIterations( parameters.getParameter< int >( prefix + "min-iterations" ) );
   this->setConvergenceResidue( parameters.getParameter< double >( prefix + "convergence-residue" ) );
   this->setDivergenceResidue( parameters.getParameter< double >( prefix + "divergence-residue" ) );
   // TODO: setting refresh rate should be done in SolverStarter::setup (it's not a parameter of the IterativeSolver)
   this->setRefreshRate( parameters.getParameter< int >( prefix + "refresh-rate" ) );
   this->residualHistoryFileName = parameters.getParameter< String >( prefix + "residual-history-file" );
   if( this->residualHistoryFileName )
      this->residualHistoryFile.open( this->residualHistoryFileName.getString() );
   return true;
}

template< typename Real, typename Index, typename SolverMonitor >
void
IterativeSolver< Real, Index, SolverMonitor >::
setMaxIterations( const Index& maxIterations )
{
   this->maxIterations = maxIterations;
}

template< typename Real, typename Index, typename SolverMonitor >
const Index&
IterativeSolver< Real, Index, SolverMonitor >::
getMaxIterations() const
{
   return this->maxIterations;
}

template< typename Real, typename Index, typename SolverMonitor >
void
IterativeSolver< Real, Index, SolverMonitor >::
setMinIterations( const Index& minIterations )
{
   this->minIterations = minIterations;
}

template< typename Real, typename Index, typename SolverMonitor >
const Index&
IterativeSolver< Real, Index, SolverMonitor >::
getMinIterations() const
{
   return this->minIterations;
}

template< typename Real, typename Index, typename SolverMonitor >
void
IterativeSolver< Real, Index, SolverMonitor >::
resetIterations()
{
   this->currentIteration = 0;
   if( this->solverMonitor )
      this->solverMonitor->setIterations( 0 );
}

template< typename Real, typename Index, typename SolverMonitor >
bool
IterativeSolver< Real, Index, SolverMonitor >::
nextIteration()
{
   // this->checkNextIteration() must be called before the iteration counter is incremented
   bool result = this->checkNextIteration();
   this->currentIteration++;
   if( this->solverMonitor )
   {
      this->solverMonitor->setIterations( this->getIterations() );
   }
   return result;
}

template< typename Real, typename Index, typename SolverMonitor >
bool
IterativeSolver< Real, Index, SolverMonitor >::
checkNextIteration()
{
   this->refreshSolverMonitor();

   if( std::isnan( this->getResidue() ) ||
       this->getIterations() > this->getMaxIterations()  ||
       ( this->getResidue() > this->getDivergenceResidue() && this->getIterations() >= this->getMinIterations() ) ||
       ( this->getResidue() < this->getConvergenceResidue() && this->getIterations() >= this->getMinIterations() ) )
      return false;
   return true;
}

template< typename Real, typename Index, typename SolverMonitor >
bool
IterativeSolver< Real, Index, SolverMonitor >::
checkConvergence()
{
   if( std::isnan( this->getResidue() ) )
   {
      std::cerr << std::endl << "The residue is NaN." << std::endl;
      return false;
   }
   if(( this->getResidue() > this->getDivergenceResidue() &&
         this->getIterations() > this->minIterations ) )
   {
      std::cerr << std::endl  << "The residue has exceeded allowed tolerance " << this->getDivergenceResidue() << "." << std::endl;
      return false;
   }
   if( this->getIterations() >= this->getMaxIterations() )
   {
      std::cerr << std::endl  << "The solver has exceeded maximal allowed number of iterations " << this->getMaxIterations() << "." << std::endl;
      return false;
   }
   if( this->getResidue() > this->getConvergenceResidue() )
   {
      std::cerr << std::endl  << "The residue ( = " << this->getResidue() << " ) is too large( > " << this->getConvergenceResidue() << " )." << std::endl;
      return false;
   }
   return true;
}

template< typename Real, typename Index, typename SolverMonitor >
const Index&
IterativeSolver< Real, Index, SolverMonitor >::
getIterations() const
{
   return this->currentIteration;
}

template< typename Real, typename Index, typename SolverMonitor >
void
IterativeSolver< Real, Index, SolverMonitor >::
setConvergenceResidue( const Real& convergenceResidue )
{
   this->convergenceResidue = convergenceResidue;
}

template< typename Real, typename Index, typename SolverMonitor >
const Real&
IterativeSolver< Real, Index, SolverMonitor >::
getConvergenceResidue() const
{
   return this->convergenceResidue;
}

template< typename Real, typename Index, typename SolverMonitor >
void
IterativeSolver< Real, Index, SolverMonitor >::
setDivergenceResidue( const Real& divergenceResidue )
{
   this->divergenceResidue = divergenceResidue;
}

template< typename Real, typename Index, typename SolverMonitor >
const Real&
IterativeSolver< Real, Index, SolverMonitor >::
getDivergenceResidue() const
{
   return this->divergenceResidue;
}

template< typename Real, typename Index, typename SolverMonitor >
void
IterativeSolver< Real, Index, SolverMonitor >::
setResidue( const Real& residue )
{
   this->currentResidue = residue;
   if( this->solverMonitor )
      this->solverMonitor->setResidue( this->getResidue() );
   if( this->residualHistoryFile ) {
      if( this->getIterations() == 0 )
         this->residualHistoryFile << "\n";
      this->residualHistoryFile << this->getIterations() << "\t" << std::scientific << residue << std::endl;
   }
}

template< typename Real, typename Index, typename SolverMonitor >
const Real&
IterativeSolver< Real, Index, SolverMonitor >::
getResidue() const
{
   return this->currentResidue;
}

// TODO: setting refresh rate should be done in SolverStarter::setup (it's not a parameter of the IterativeSolver)
template< typename Real, typename Index, typename SolverMonitor >
void
IterativeSolver< Real, Index, SolverMonitor >::
setRefreshRate( const Index& refreshRate )
{
   this->refreshRate = refreshRate;
}

template< typename Real, typename Index, typename SolverMonitor >
void
IterativeSolver< Real, Index, SolverMonitor >::
setSolverMonitor( SolverMonitorType& solverMonitor )
{
   this->solverMonitor = &solverMonitor;
}

template< typename Real, typename Index, typename SolverMonitor >
void
IterativeSolver< Real, Index, SolverMonitor >::
refreshSolverMonitor( bool force )
{
   if( this->solverMonitor )
   {
      this->solverMonitor->setIterations( this->getIterations() );
      this->solverMonitor->setResidue( this->getResidue() );
      this->solverMonitor->setRefreshRate( this-> refreshRate );
   }
}

} // namespace Solvers
} // namespace TNL
