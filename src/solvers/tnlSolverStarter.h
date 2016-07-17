/***************************************************************************
                          tnlSolverStarter.h  -  description
                             -------------------
    begin                : Mar 9, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLSOLVERSTARTER_H_
#define TNLSOLVERSTARTER_H_

#include <config/tnlParameterContainer.h>
#include <core/tnlTimer.h>
#include <ostream>

template< typename MeshConfig >
class tnlSolverStarter
{
   public:

   tnlSolverStarter();

   template< typename Problem >
   static bool run( const tnlParameterContainer& parameters );

   template< typename Solver >
   bool writeEpilog( ostream& str, const Solver& solver );

   template< typename Problem, typename TimeStepper >
   bool runPDESolver( Problem& problem,
                      const tnlParameterContainer& parameters,
                      TimeStepper& timeStepper );

   protected:

   int logWidth;

   tnlTimer ioTimer, computeTimer, totalTimer;
};

#include <solvers/tnlSolverStarter_impl.h>

#endif /* TNLSOLVERSTARTER_H_ */
