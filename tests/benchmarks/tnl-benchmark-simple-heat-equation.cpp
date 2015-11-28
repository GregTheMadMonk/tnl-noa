/***************************************************************************
                          tnl-benchmark-simple-heat-equation.cpp  -  description
                             -------------------
    begin                : Nov 28, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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

#include "tnl-benchmark-simple-heat-equation.h"
#include <stdio.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>


int main( int argc, char* argv[] )
{
   tnlConfigDescription config;
   config.template addEntry< int >( "grid-x-size", "Grid size along x-axis.", 100 );
   config.template addEntry< int >( "grid-y-size", "Grid size along y-axis.", 100 );
   config.template addEntry< double >( "domain-x-size", "Domain size along x-axis.", 2.0 );
   config.template addEntry< double >( "domain-y-size", "Domain size along y-axis.", 2.0 );
   config.template addEntry< double >( "sigma", "Sigma in exponential initial condition.", 2.0 );
   config.template addEntry< double >( "time-step", "Time step. By default it is proportional to one over space step square.", 0.0 );
   config.template addEntry< double >( "final-time", "Final time of the simulation.", 1.0 );
   config.template addEntry< bool >( "verbose", "Verbose mode.", true );
   
   tnlParameterContainer parameters;
   if( ! parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;
   
   if( ! solveHeatEquation< double, int >( parameters  ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;   
}
