// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>

namespace TNL {
namespace Solvers {
namespace ODE {

template< typename Real, typename Index >
void
StaticExplicitSolver< Real, Index >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   //IterativeSolver< typename Problem::RealType, typename Problem::IndexType >::configSetup( config, prefix );
}

template< typename Real, typename Index >
bool
StaticExplicitSolver< Real, Index >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   this->setVerbose( parameters.getParameter< int >( "verbose" ) );
   return IterativeSolver< RealType, IndexType, SolverMonitor >::setup( parameters, prefix );
}

template< typename Real, typename Index >
__cuda_callable__ void
StaticExplicitSolver< Real, Index >::
setTime( const RealType& time )
{
   this->time = time;
};

template< typename Real, typename Index >
__cuda_callable__ const Real&
StaticExplicitSolver< Real, Index >::
getTime() const
{
   return this->time;
};

template< typename Real, typename Index >
__cuda_callable__ void
StaticExplicitSolver< Real, Index >::
setTau( const RealType& tau )
{
   this->tau = tau;
};

template< typename Real, typename Index >
__cuda_callable__ const Real&
StaticExplicitSolver< Real, Index >::
getTau() const
{
   return this->tau;
};

template< typename Real, typename Index >
__cuda_callable__ void
StaticExplicitSolver< Real, Index >::
setMaxTau( const RealType& maxTau )
{
   this->maxTau = maxTau;
};


template< typename Real, typename Index >
__cuda_callable__ const Real&
StaticExplicitSolver< Real, Index >::
getMaxTau() const
{
   return this->maxTau;
};


template< typename Real, typename Index >
__cuda_callable__ const Real&
StaticExplicitSolver< Real, Index >::
getStopTime() const
{
    return this->stopTime;
}

template< typename Real, typename Index >
__cuda_callable__ void
StaticExplicitSolver< Real, Index >::
setStopTime( const RealType& stopTime )
{
    this->stopTime = stopTime;
}

template< typename Real, typename Index >
__cuda_callable__ void
StaticExplicitSolver< Real, Index >::
setVerbose( IndexType v )
{
   this->verbosity = v;
};

template< typename Real, typename Index >
void
StaticExplicitSolver< Real, Index >::
setTestingMode( bool testingMode )
{
   this->testingMode = testingMode;
}

} // namespace ODE
} // namespace Solvers
} // namespace TNL