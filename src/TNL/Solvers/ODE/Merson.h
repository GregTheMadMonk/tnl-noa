// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <math.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Solvers/ODE/ExplicitSolver.h>

namespace TNL {
namespace Solvers {
namespace ODE {

template< class Vector,
          typename SolverMonitor = IterativeSolverMonitor< typename Vector::RealType, typename Vector::IndexType > >
class Merson : public ExplicitSolver< typename Vector::RealType, typename Vector::IndexType, SolverMonitor >
{
public:
   using ProblemType = Problem;
   using DofVectorType = typename Problem::DofVectorType;
   using RealType = typename Problem::RealType;
   using DeviceType = typename Problem::DeviceType;
   using IndexType = typename Problem::IndexType;
   using DofVectorPointer = Pointers::SharedPointer< DofVectorType, DeviceType >;
   using SolverMonitorType = SolverMonitor;

      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType  = typename Vector::IndexType;
      using VectorType = Vector;
      using DofVectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
      using SolverMonitorType = SolverMonitor;

      Merson() = default;

   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" );

   bool
   solve( DofVectorPointer& u );

      template< typename RHSFunction >
      bool solve( VectorType& u, RHSFunction&& rhs );

   DofVectorPointer _k1, _k2, _k3, _k4, _k5, _kAux;

      void writeGrids( const DofVectorType& u );

      DofVectorType _k1, _k2, _k3, _k4, _k5, _kAux;

      /****
       * This controls the accuracy of the solver
       */
      RealType adaptivity = 0.00001;
};

}  // namespace ODE
}  // namespace Solvers
}  // namespace TNL

#include <TNL/Solvers/ODE/Merson.hpp>
