// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <TNL/Solvers/IterativeSolverMonitor.h>
#include <TNL/Solvers/IterativeSolver.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Containers/Vector.h>

namespace TNL {
namespace Solvers {
namespace ODE {

template< typename Real = double,
          typename Index = int,
          typename SolverMonitor = IterativeSolverMonitor< Real, Index > >
class ExplicitSolver : public IterativeSolver< Real, Index, SolverMonitor >
{
   public:

   using RealType = Real;
   using IndexType = Index;
   using SolverMonitorType = SolverMonitor;

   ExplicitSolver();

   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" );

   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

   void setTime( const RealType& t );

   const RealType&
   getTime() const;

   void
   setStopTime( const RealType& stopTime );

   const RealType& getStopTime() const;

   void
   setTau( const RealType& tau );

   const RealType&
   getTau() const;

   void
   setMaxTau( const RealType& maxTau );

   const RealType&
   getMaxTau() const;

   void
   setVerbose( IndexType v );

   void setTestingMode( bool testingMode );

   void
   setRefreshRate( const IndexType& refreshRate );

   void
   refreshSolverMonitor( bool force = false );

protected:
   /****
    * Current time of the parabolic problem.
    */
   RealType time;

   /****
    * The method solve will stop when reaching the stopTime.
    */
   RealType stopTime;

   /****
    * Current time step.
    */
   RealType tau;

   RealType maxTau;

   IndexType verbosity;

   bool testingMode;
};

}  // namespace ODE
}  // namespace Solvers
}  // namespace TNL

#include <TNL/Solvers/ODE/ExplicitSolver.hpp>
