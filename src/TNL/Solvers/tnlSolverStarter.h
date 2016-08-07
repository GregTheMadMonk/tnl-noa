/***************************************************************************
                          tnlSolverStarter.h  -  description
                             -------------------
    begin                : Mar 9, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Timer.h>
#include <ostream>

namespace TNL {

template< typename MeshConfig >
class tnlSolverStarter
{
   public:

   tnlSolverStarter();

   template< typename Problem >
   static bool run( const Config::ParameterContainer& parameters );

   template< typename Solver >
   bool writeEpilog( std::ostream& str, const Solver& solver );

   template< typename Problem, typename TimeStepper >
   bool runPDESolver( Problem& problem,
                      const Config::ParameterContainer& parameters,
                      TimeStepper& timeStepper );

   protected:

   int logWidth;

   tnlTimer ioTimer, computeTimer, totalTimer;
};

} // namespace TNL

#include <TNL/Solvers/tnlSolverStarter_impl.h>
