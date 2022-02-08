// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/StaticVector.h>
#include <TNL/Solvers/ODE/StaticEuler.h>

namespace TNL {
namespace Solvers {
namespace ODE {

/////
// Specialization of the Euler solver for numeric types
template< typename Real >
void
StaticEuler< Real >::
configSetup( Config::ConfigDescription& config, const String& prefix )
{
   config.addEntry< double >( prefix + "euler-cfl", "Coefficient C in the Courant–Friedrichs–Lewy condition.", 0.0 );
};

template< typename Real >
bool
StaticEuler< Real >::
setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   StaticExplicitSolver< RealType, IndexType >::setup( parameters, prefix );
   if( parameters.checkParameter( prefix + "euler-cfl" ) )
      this->setCFLCondition( parameters.getParameter< double >( prefix + "euler-cfl" ) );
   return true;
}

template< typename Real >
__cuda_callable__ void
StaticEuler< Real >::
setCFLCondition( const RealType& cfl )
{
   this -> cflCondition = cfl;
}

template< typename Real >
__cuda_callable__  auto
StaticEuler< Real >::
getCFLCondition() const -> const RealType&
{
   return this -> cflCondition;
}

template< typename Real >
   template< typename RHSFunction >
__cuda_callable__ bool
StaticEuler< Real >::
solve( VectorType& u, RHSFunction&& rhsFunction )
{
   /////
   // First setup the supporting coefficient k1.
   k1 = 0.0;

   /////
   // Set necessary parameters
   RealType& time = this->time;
   RealType currentTau = min( this->getTau(), this->getMaxTau() );
   if( time + currentTau > this->getStopTime() ) currentTau = this->getStopTime() - time;
   if( currentTau == 0.0 ) return true;
   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   /////
   // Start the main loop
   while( 1 )
   {
      /////
      // Compute the RHS
      rhsFunction( time, currentTau, u, k1 );

      RealType lastResidue = this->getResidue();
      RealType maxResidue( 0.0 );
      if( this->cflCondition != 0.0 ) {
         maxResidue = abs( k1 );
         if( currentTau * maxResidue > this->cflCondition ) {
            currentTau *= 0.9;
            continue;
         }
      }
      u += currentTau * k1;
      this->setResidue( abs( k1 ) );

      /////
      // When time is close to stopTime the new residue may be inaccurate significantly.
      if( currentTau + time == this->stopTime ) this->setResidue( lastResidue );
      time += currentTau;
      //this->problem->applyBoundaryConditions( time, _u );

      if( ! this->nextIteration() )
         return this->checkConvergence();

      /////
      // Compute the new time step.
      if( time + currentTau > this -> getStopTime() )
         currentTau = this -> getStopTime() - time; //we don't want to keep such tau
      else this -> tau = currentTau;

      /////
      // Check stop conditions.
      if( time >= this->getStopTime() ||
          ( this -> getConvergenceResidue() != 0.0 && this->getResidue() < this -> getConvergenceResidue() ) )
         return true;

      if( this -> cflCondition != 0.0 )
      {
         currentTau /= 0.95;
         currentTau = min( currentTau, this->getMaxTau() );
      }
   }
   return false; // just to avoid warnings
};

////
// Specialization for static vectors
template< int Size_, typename Real >
void
StaticEuler< Containers::StaticVector< Size_, Real > >::
configSetup( Config::ConfigDescription& config, const String& prefix )
{
   config.addEntry< double >( prefix + "euler-cfl", "Coefficient C in the Courant–Friedrichs–Lewy condition.", 0.0 );
};

template< int Size_, typename Real >
bool
StaticEuler< Containers::StaticVector< Size_, Real > >::
setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   StaticExplicitSolver< RealType, IndexType >::setup( parameters, prefix );
   if( parameters.checkParameter( prefix + "euler-cfl" ) )
      this->setCFLCondition( parameters.getParameter< double >( prefix + "euler-cfl" ) );
   return true;
}

template< int Size_, typename Real >
__cuda_callable__ void
StaticEuler< Containers::StaticVector< Size_, Real > >::
setCFLCondition( const RealType& cfl )
{
   this -> cflCondition = cfl;
}

template< int Size_, typename Real >
__cuda_callable__  auto
StaticEuler< Containers::StaticVector< Size_, Real > >::
getCFLCondition() const -> const RealType&
{
   return this -> cflCondition;
}

template< int Size_, typename Real >
   template< typename RHSFunction >
__cuda_callable__ bool
StaticEuler< Containers::StaticVector< Size_, Real > >::
solve( VectorType& u, RHSFunction&& rhsFunction )
{
   /////
   // First setup the supporting vector k1.
   k1 = 0.0;

   /////
   // Set necessary parameters
   RealType& time = this->time;
   RealType currentTau = min( this->getTau(), this->getMaxTau() );
   if( time + currentTau > this->getStopTime() ) currentTau = this->getStopTime() - time;
   if( currentTau == 0.0 ) return true;
   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   /////
   // Start the main loop
   while( 1 )
   {
      /////
      // Compute the RHS
      rhsFunction( time, currentTau, u, k1 );

      RealType lastResidue = this->getResidue();
      RealType maxResidue( 0.0 );
      if( this->cflCondition != 0.0 ) {
         maxResidue = max( abs( k1 ) );
         if( currentTau * maxResidue > this->cflCondition ) {
            currentTau *= 0.9;
            continue;
         }
      }
      this->setResidue( addAndReduceAbs( u, currentTau * k1, TNL::Plus(), ( RealType ) 0.0 ) / ( currentTau * ( RealType ) u.getSize() ) );

      /////
      // When time is close to stopTime the new residue may be inaccurate significantly.
      if( currentTau + time == this->stopTime ) this->setResidue( lastResidue );
      time += currentTau;
      //this->problem->applyBoundaryConditions( time, _u );

      if( ! this->nextIteration() )
         return this->checkConvergence();

      /////
      // Compute the new time step.
      if( time + currentTau > this -> getStopTime() )
         currentTau = this -> getStopTime() - time; //we don't want to keep such tau
      else this -> tau = currentTau;

      /////
      // Check stop conditions.
      if( time >= this->getStopTime() ||
          ( this -> getConvergenceResidue() != 0.0 && this->getResidue() < this -> getConvergenceResidue() ) )
         return true;

      if( this -> cflCondition != 0.0 )
      {
         currentTau /= 0.95;
         currentTau = min( currentTau, this->getMaxTau() );
      }
   }
   return false; // just to avoid warnings
};

} // namespace ODE
} // namespace Solvers
} // namespace TNL
