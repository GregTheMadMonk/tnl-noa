/***************************************************************************
                          SolverStarter.h  -  description
                             -------------------
    begin                : Mar 9, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Timer.h>
#include <TNL/Solvers/SolverMonitor.h>
#include <ostream>

namespace TNL {
namespace Solvers {

template< typename ConfigTag >
class SolverStarter
{
   public:

   SolverStarter();

   template< typename Problem >
   static bool run( const Config::ParameterContainer& parameters );

   template< typename Solver >
   bool writeEpilog( std::ostream& str, const Solver& solver );

   template< typename Problem, typename TimeStepper >
   bool runPDESolver( Problem& problem,
                      const Config::ParameterContainer& parameters );

   protected:

   int logWidth;

   Timer ioTimer, computeTimer, totalTimer;
};

} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/SolverStarter.hpp>
