// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Solvers/ODE/ExplicitSolver.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/StaticVector.h>

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

   Euler() = default;

   static void configSetup( Config::ConfigDescription& config, const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" );

   void setCourantNumber( const RealType& cfl );

   const RealType& getCourantNumber() const;

   template< typename RHSFunction >
   bool solve( VectorType& u, RHSFunction&& rhs );

protected:
   DofVectorType _k1;

   RealType courantNumber = 0.0;
};

} // namespace ODE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/ODE/Euler.hpp>
