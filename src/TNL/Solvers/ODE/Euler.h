// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Solvers/ODE/ExplicitSolver.h>
#include <TNL/Config/ParameterContainer.h>

namespace TNL {
namespace Solvers {
namespace ODE {

template< typename Vector,
          typename SolverMonitor = IterativeSolverMonitor< typename Vector::RealType, typename Vector::IndexType > >
class Euler : public ExplicitSolver< typename Vector::RealType, typename Vector::IndexType, SolverMonitor >
{
public:
   using RealType = typename Vector::RealType;
   using DeviceType = typename Vector::DeviceType;
   using IndexType  = typename Vector::IndexType;
   using VectorType = Vector;
   using DofVectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using SolverMonitorType = SolverMonitor;

   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" );

   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" );

   const RealType&
   getCFLCondition() const;

   bool
   solve( DofVectorPointer& u );

      template< typename RHSFunction >
      bool solve( VectorType& u, RHSFunction&& rhs );

   protected:
      DofVectorType _k1;

      RealType cflCondition;
};

}  // namespace ODE
}  // namespace Solvers
}  // namespace TNL

#include <TNL/Solvers/ODE/Euler.hpp>
