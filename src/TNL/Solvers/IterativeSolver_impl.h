/***************************************************************************
                          IterativeSolver_impl.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <cmath>
#include <float.h>

namespace TNL {
namespace Solvers {   

template< typename Real, typename Index >
IterativeSolver< Real, Index> :: IterativeSolver()
: maxIterations( 100000 ),
  minIterations( 0 ),
  currentIteration( 0 ),
  convergenceResidue( 1.0e-6 ),
  divergenceResidue( DBL_MAX ),
  currentResidue( 0 ),
  solverMonitor( 0 ),
  refreshRate( 1 )
{
};

template< typename Real, typename Index >
void IterativeSolver< Real, Index> :: configSetup( Config::ConfigDescription& config,
                                                      const String& prefix )
{
   config.addEntry< int >   ( prefix + "max-iterations", "Maximal number of iterations the solver may perform.", 10000000000 );
   config.addEntry< int >   ( prefix + "min-iterations", "Minimal number of iterations the solver must perform.", 0 );
   config.addEntry< double >( prefix + "convergence-residue", "Convergence occurs when the residue drops bellow this limit.", 1.0e-6 );
   config.addEntry< double >( prefix + "divergence-residue", "Divergence occurs when the residue exceeds given limit.", DBL_MAX );
   config.addEntry< int >   ( prefix + "refresh-rate", "Number of iterations between solver monitor refreshes.", 1 );
}

template< typename Real, typename Index >
bool IterativeSolver< Real, Index> :: setup( const Config::ParameterContainer& parameters,
                                               const String& prefix )
{
   this->setMaxIterations( parameters.getParameter< int >( "max-iterations" ) );
   this->setMinIterations( parameters.getParameter< int >( "min-iterations" ) );
   this->setConvergenceResidue( parameters.getParameter< double >( "convergence-residue" ) );
   this->setDivergenceResidue( parameters.getParameter< double >( "divergence-residue" ) );
   this->setRefreshRate( parameters.getParameter< int >( "refresh-rate" ) );
   return true;
}

template< typename Real, typename Index >
void IterativeSolver< Real, Index> :: setMaxIterations( const Index& maxIterations )
{
   this->maxIterations = maxIterations;
}

template< typename Real, typename Index >
const Index& IterativeSolver< Real, Index> :: getMaxIterations() const
{
   return this->maxIterations;
}

template< typename Real, typename Index >
void IterativeSolver< Real, Index> :: setMinIterations( const Index& minIterations )
{
   this->minIterations = minIterations;
}

template< typename Real, typename Index >
const Index& IterativeSolver< Real, Index> :: getMinIterations() const
{
   return this->minIterations;
}

template< typename Real, typename Index >
void IterativeSolver< Real, Index> :: resetIterations()
{
   this->currentIteration = 0;
}

template< typename Real, typename Index >
bool IterativeSolver< Real, Index> :: nextIteration()
{
   // this->checkNextIteration() must be called before the iteration counter is incremented
   bool result = this->checkNextIteration();
   this->currentIteration++;
   return result;
}

template< typename Real, typename Index >
bool IterativeSolver< Real, Index> :: checkNextIteration()
{
   // TODO: fix
   //Assert( solverMonitor, );
   if( this->solverMonitor )
   {
      solverMonitor->setIterations( this->currentIteration );
      solverMonitor->setResidue( this->getResidue() );
      if( this->currentIteration % this->refreshRate == 0 )
         solverMonitor->refresh();
   }

   if( std::isnan( this->getResidue() ) ||
       this->getIterations() > this->getMaxIterations()  ||
       ( this->getResidue() > this->getDivergenceResidue() && this->getIterations() >= this->getMinIterations() ) ||
       ( this->getResidue() < this->getConvergenceResidue() && this->getIterations() >= this->getMinIterations() ) )
      return false;
   return true;
}

template< typename Real, typename Index >
bool
IterativeSolver< Real, Index>::
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

template< typename Real, typename Index >
const Index&
IterativeSolver< Real, Index>::
getIterations() const
{
   return this->currentIteration;
}

template< typename Real, typename Index >
void IterativeSolver< Real, Index> :: setConvergenceResidue( const Real& convergenceResidue )
{
   this->convergenceResidue = convergenceResidue;
}

template< typename Real, typename Index >
const Real& IterativeSolver< Real, Index> :: getConvergenceResidue() const
{
   return this->convergenceResidue;
}

template< typename Real, typename Index >
void IterativeSolver< Real, Index> :: setDivergenceResidue( const Real& divergenceResidue )
{
   this->divergenceResidue = divergenceResidue;
}

template< typename Real, typename Index >
const Real& IterativeSolver< Real, Index> :: getDivergenceResidue() const
{
   return this->divergenceResidue;
}


template< typename Real, typename Index >
void IterativeSolver< Real, Index> :: setResidue( const Real& residue )
{
   this->currentResidue = residue;
}

template< typename Real, typename Index >
const Real& IterativeSolver< Real, Index> :: getResidue() const
{
   return this->currentResidue;
}

template< typename Real, typename Index >
void IterativeSolver< Real, Index> :: setRefreshRate( const Index& refreshRate )
{
   this->refreshRate = refreshRate;
}

template< typename Real, typename Index >
void IterativeSolver< Real, Index> :: setSolverMonitor( IterativeSolverMonitor< Real, Index >& solverMonitor )
{
   this->solverMonitor = &solverMonitor;
}

template< typename Real, typename Index >
void IterativeSolver< Real, Index> :: refreshSolverMonitor( bool force )
{
   if( this->solverMonitor )
   {
      this->solverMonitor -> setIterations( this->getIterations() );
      this->solverMonitor -> setResidue( this->getResidue() );
      this->solverMonitor -> setRefreshRate( this-> refreshRate );
      this->solverMonitor -> refresh( force );
   }
}


#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

extern template class IterativeSolver< float,  int >;
extern template class IterativeSolver< double, int >;
extern template class IterativeSolver< float,  long int >;
extern template class IterativeSolver< double, long int >;

#ifdef HAVE_CUDA
extern template class IterativeSolver< float,  int >;
extern template class IterativeSolver< double, int >;
extern template class IterativeSolver< float,  long int >;
extern template class IterativeSolver< double, long int >;
#endif

#endif

} // namespace Solvers
} // namespace TNL
