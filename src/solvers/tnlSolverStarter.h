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

#pragma once

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
   bool runTimeDependentPDESolver( Problem& problem,
                                   const tnlParameterContainer& parameters,
                                   TimeStepper& timeStepper );
   
   template< typename Problem >
   bool runTimeIndependentPDESolver( Problem& problem,
                                     const tnlParameterContainer& parameters );

   protected:

   int logWidth;

   tnlTimer ioTimer, computeTimer, totalTimer;
};

#include <solvers/tnlSolverStarter_impl.h>
