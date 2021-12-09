/***************************************************************************
                          PDESolver.h  -  description
                             -------------------
    begin                : Nov 11, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Timer.h>
#include <TNL/Logger.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Solvers/IterativeSolverMonitor.h>

namespace TNL {
namespace Solvers {
namespace PDE {

template< typename Real,
          typename Index >
class PDESolver
{
   public:
      using RealType = Real;
      using IndexType = Index;
      using SolverMonitorType = IterativeSolverMonitor< RealType, IndexType >;


      PDESolver();

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" );

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" );

      bool writeProlog( Logger& logger,
                        const Config::ParameterContainer& parameters );

      void setIoTimer( Timer& ioTimer);

      void setComputeTimer( Timer& computeTimer );

      void setTotalTimer( Timer& totalTimer );

      void setSolverMonitor( SolverMonitorType& solverMonitor );

      SolverMonitorType& getSolverMonitor();

      bool writeEpilog( Logger& logger ) const;

   protected:

      Timer *ioTimer, *computeTimer, *totalTimer;

      SolverMonitorType *solverMonitorPointer;
};

} // namespace PDE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/PDE/PDESolver.hpp>
