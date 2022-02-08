// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <TNL/Solvers/StaticIterativeSolver.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/StaticVector.h>

namespace TNL {
namespace Solvers {
namespace ODE {

template< typename Real = double,
          typename Index = int >
class StaticExplicitSolver : public StaticIterativeSolver< Real, Index >
{
   public:

   using RealType = Real;
   using IndexType = Index;

   __cuda_callable__
   StaticExplicitSolver() = default;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

   __cuda_callable__
   void setTime( const RealType& t );

   __cuda_callable__
   const RealType& getTime() const;

   __cuda_callable__
   void setStopTime( const RealType& stopTime );

   __cuda_callable__
   const RealType& getStopTime() const;

   __cuda_callable__
   void setTau( const RealType& tau );

   __cuda_callable__
   const RealType& getTau() const;

   __cuda_callable__
   void setMaxTau( const RealType& maxTau );

   __cuda_callable__
   const RealType& getMaxTau() const;

   __cuda_callable__
   void setVerbose( IndexType v );

   __cuda_callable__
   void setTestingMode( bool testingMode );

protected:

   /****
    * Current time of the parabolic problem.
    */
   RealType time = 0.0;

   /****
    * The method solve will stop when reaching the stopTime.
    */
   RealType stopTime;

   /****
    * Current time step.
    */
   RealType tau = 0.0;

   RealType maxTau = std::numeric_limits< RealType >::max();

   IndexType verbosity = 0;

   bool testingMode = false;
};

} // namespace ODE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/ODE/StaticExplicitSolver.hpp>
