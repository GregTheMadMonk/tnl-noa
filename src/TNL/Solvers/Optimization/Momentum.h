// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Solvers/IterativeSolver.h>

namespace TNL {
   namespace Solvers {
      namespace Optimization {

template< typename Vector, typename SolverMonitor =  IterativeSolverMonitor< typename Vector::RealType, typename Vector::IndexType > >
class Momentum : public IterativeSolver< typename Vector::RealType, typename Vector::IndexType, SolverMonitor >
{
public:
   using RealType = typename Vector::RealType;
   using DeviceType = typename Vector::DeviceType;
   using IndexType = typename Vector::IndexType;
   using VectorType = Vector;
   using VectorView = typename Vector::ViewType;

   Momentum() = default;

   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" );

   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

   void
   setRelaxation( const RealType& lambda );

   const RealType&
   getRelaxation() const;

   void
   setMomentum( const RealType& beta );

   const RealType&
   getMomentum() const;

   template< typename GradientGetter >
   bool
   solve( VectorView& w, GradientGetter&& getGradient );

protected:

   RealType relaxation = 1.0, momentum = 0.9;

   VectorType gradient, v;

};

      } //namespace Optimization
   } //namespace Solvers
} //namespace TNL

#include <TNL/Solvers/Optimization/Momentum.hpp>
