/***************************************************************************
                          main.cpp  -  description
                             -------------------
    begin                : Jan 12, 2013
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

#include "simple-solver-conf.h"
#include "simpleProblemSetter.h"
#include <solvers/tnlSolver.h>
#include <config/tnlConfigDescription.h>

void configSetup( tnlConfigDescription& config )
{
   config.addDelimiter( "Simple solver settings:" );
   config.addEntry        < tnlString > ( "problem-name", "This defines particular problem.", "simpl" );
   config.addEntry        < tnlString >( "mesh", "A file containing the numerical mesh.", "mesh.tnl" );
   config.addRequiredEntry< tnlString >( "time-discretisation", "Time discretisation for the time dependent problems." );
      config.addEntryEnum( "explicit" );
      config.addEntryEnum( "semi-implicit" );
      config.addEntryEnum( "fully-implicit" );
   config.addEntry        < tnlString >( "real-type", "Precision of the floating point arithmetics.", "double" );
      config.addEntryEnum( "float" );
      config.addEntryEnum( "double" );
      config.addEntryEnum( "long-double" );
   config.addEntry        < tnlString >( "index-type", "Indexing type for arrays, vectors, matrices etc.", "int" );
      config.addEntryEnum( "int" );
      config.addEntryEnum( "long-int" );
   config.addEntry        < tnlString >( "device", "Device to use for the computations.", "host" );
      config.addEntryEnum( "host" );
      config.addEntryEnum( "cuda" );
   config.addRequiredEntry< tnlString >( "discrete-solver", "The main solver for the discretised problem." );
   config.addEntry< double >( "merson-adaptivity", "Parameter controling adaptivity of the Runge-Kutta-Merson solver.", 1.0e-4 );
      //config.setEntryMinimum< double >( 0.0 );
   config.addEntry< double >( "initial-tau", "Initial time step for the time dependent PDE solver. It can be changed if an adaptive solver is used.", 1.0e-5 );
      //config.setEntryMinimum( 0.0 );
   config.addRequiredEntry< double >( "snapshot-period", "Time period for writing a state of the time dependent problem to a file." );
      //config.setEntryMinimum( 0.0 );
   config.addRequiredEntry< double >( "final-time", "Stop time of the time dependent simulation." );
      //config.setEntryMinimum( 0.0 );
   config.addEntry< int >( "verbose", "Set the verbose mode. The higher number the more messages are generated.", 1 );
   config.addEntry< tnlString >( "log-file", "File name for the log.", "simple-problem.log" );
}

int main( int argc, char* argv[] )
{
   tnlSolver< simpleProblemSetter > solver;
   if( ! solver. run( CONFIG_FILE, argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}


