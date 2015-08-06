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

template< typename ConfigTag >
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

   tnlTimerRT ioRtTimer, computeRtTimer, totalRtTimer;

   tnlTimerCPU ioCpuTimer, computeCpuTimer, totalCpuTimer;
};

#include <solvers/tnlSolverStarter_impl.h>

#endif /* TNLSOLVERSTARTER_H_ */
