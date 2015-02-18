/***************************************************************************
                          main.cpp  -  description
                             -------------------
    begin                : Jul 8 , 2014
    copyright            : (C) 2014 by Tomas Sobotik
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "tnlParallelEikonalSolver.h"
#include "parallelEikonalConfig.h"
#include "MainBuildConfig.h"
#include <solvers/tnlConfigTags.h>
#include <operators/godunov-eikonal/parallelGodunovEikonal.h>
#include <mesh/tnlGrid.h>

typedef MainBuildConfig BuildConfig;

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription configDescription;
   parallelEikonalConfig< BuildConfig >::configSetup( configDescription );

   if( ! ParseCommandLine( argc, argv, configDescription, parameters ) )
      return false;

   //if (parameters.GetParameter <tnlString>("scheme") == "godunov")
   //{
	   typedef parallelGodunovEikonalScheme< tnlGrid<2,double,tnlHost, int>, double, int > SchemeType;
   	   tnlParallelEikonalSolver<SchemeType> solver;
   	   if(!solver.init(parameters))
   	   {
   		   cerr << "Solver failed to initialize." << endl;
   		   return EXIT_FAILURE;
   	   }
   	   cout << "-------------------------------------------------------------" << endl;
   	   cout << "Starting solver loop..." << endl;
   	   solver.run();
  // }

   return EXIT_SUCCESS;
}


