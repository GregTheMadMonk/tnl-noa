/***************************************************************************
                          Timer.h  -  description
                             -------------------
    begin                : Mar 14, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Solvers/Optimization/GradientDescent.h>

namespace TNL {
   namespace Solvers {
      namespace Optimization {


template< typename Vector, typename SolverMonitor >
void
GradientDescent< Vector, SolverMonitor >::
configSetup( Config::ConfigDescription& config, const String& prefix )
{
   IterativeSolver< RealType, IndexType, SolverMonitor >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "gd-relaxation", "Relaxation parameter for the gradient descent.", 1.0 )
}

template< typename Vector, typename SolverMonitor >
bool
GradientDescent< Vector, SolverMonitor >::
setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   this->setRelaxation( parameters.getParameter< double >( prefix + "gd-relaxation" ) );
   return IterativeSolver< RealType, IndexType, SolverMonitor >::setup( parameters, prefix );
}

template< typename Vector, typename SolverMonitor >
void
GradientDescent< Vector, SolverMonitor >::
setRelaxation( const RealType& lambda )
{
   this->relaxation = lambda;
}

template< typename Vector, typename SolverMonitor >
auto
GradientDescent< Vector, SolverMonitor >::
getRelaxation() const -> RealType&
{
   return this->relaxation;
}

template< typename Vector, typename SolverMonitor >
   template< typename GradientGetter >
bool
GradientDescent< Vector, SolverMonitor >::
solve( VectorView& w, GradientGetter&& getGradient )
{
   this->aux.setLike( w );
   auto aux_view = aux.getView();
   auto w_view = w.getView();
   aux = 0.0;

   /////
   // Set necessary parameters
   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   /////
   // Start the main loop
   while( 1 )
   {
      /////
      // Compute the gradient
      getGradient( w_view, aux_view );
      RealType lastResidue = this->getResidue();
      this->setResidue( addAndReduceAbs( w_view, this->relaxation * aux_view, TNL::Plus(), ( RealType ) 0.0 ) / ( this->relaxation * ( RealType ) w.getSize() ) );

      if( ! this->nextIteration() )
         return this->checkConvergence();

      /////
      // Check the stop condition
      if( this -> getConvergenceResidue() != 0.0 && this->getResidue() < this -> getConvergenceResidue() )
         return true;
   }
   return false; // just to avoid warnings
}

      } //namespace Optimization
   } //namespace Solvers
} //namespace TNL
