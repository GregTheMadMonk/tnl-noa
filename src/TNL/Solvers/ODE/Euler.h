/***************************************************************************
                          Euler.h  -  description
                             -------------------
    begin                : 2008/04/01
    copyright            : (C) 2008 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Solvers/ODE/ExplicitSolver.h>
#include <TNL/Config/ParameterContainer.h>

namespace TNL {
namespace Solvers {
namespace ODE {

template< typename Problem,
          typename SolverMonitor = IterativeSolverMonitor< typename Problem::RealType, typename Problem::IndexType > >
class Euler : public ExplicitSolver< Problem, SolverMonitor >
{
   public:

      using ProblemType = Problem;
      using DofVectorType = typename ProblemType::DofVectorType;
      using RealType = typename ProblemType::RealType;
      using DeviceType = typename ProblemType::DeviceType;
      using IndexType  = typename ProblemType::IndexType;
      using DofVectorPointer = Pointers::SharedPointer<  DofVectorType, DeviceType >;
      using SolverMonitorType = SolverMonitor;

      Euler();

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" );

      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );

      void setCFLCondition( const RealType& cfl );

      const RealType& getCFLCondition() const;

      bool solve( DofVectorPointer& u );

   protected:
      DofVectorPointer _k1;

      RealType cflCondition;
};

} // namespace ODE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/ODE/Euler.hpp>
