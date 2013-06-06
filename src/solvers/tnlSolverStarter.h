/***************************************************************************
                          tnlSolverStarter.h  -  description
                             -------------------
    begin                : Mar 9, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLSOLVERSTARTER_H_
#define TNLSOLVERSTARTER_H_

#include <config/tnlParameterContainer.h>
#include <core/tnlTimerRT.h>
#include <core/tnlTimerCPU.h>
#include <ostream>

class tnlSolverStarter
{
   public:

   tnlSolverStarter();

   template< typename Problem >
   bool run( const tnlParameterContainer& parameters );

   template< typename Problem >
   bool setDiscreteSolver( Problem& problem,
                           const tnlParameterContainer& parameters );

   template< typename Problem,
             template < typename > class DiscreteSolver >
   bool setExplicitTimeDiscretisation( Problem& problem,
                                       const tnlParameterContainer& parameters,
                                       DiscreteSolver< Problem >& solver );

   template< typename Problem,
             typename DiscreteSolver >
   bool setSemiImplicitTimeDiscretisation( Problem& problem,
                                           const tnlParameterContainer& parameters,
                                           DiscreteSolver& solver);

   template< typename Problem >
   bool writeProlog( ostream& str,
                     const tnlParameterContainer& parameters,
                     const Problem& problem );

   template< typename Problem, typename TimeStepper >
   bool runPDESolver( Problem& problem,
                      const tnlParameterContainer& parameters,
                      TimeStepper& timeStepper );

   bool writeEpilog( ostream& str );

   protected:

   template< typename IterativeSolver >
   bool setIterativeSolver( IterativeSolver& solver,
                            const tnlParameterContainer& parameters ) const;

   int verbose;

   int logWidth;

   tnlTimerRT ioRtTimer, computeRtTimer, totalRtTimer;

   tnlTimerCPU ioCpuTimer, computeCpuTimer, totalCpuTimer;
};

#include <implementation/solvers/tnlSolverStarter_impl.h>

#endif /* TNLSOLVERSTARTER_H_ */
