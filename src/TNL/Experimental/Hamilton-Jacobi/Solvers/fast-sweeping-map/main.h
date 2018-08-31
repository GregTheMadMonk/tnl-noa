/***************************************************************************
                          main.h  -  description
                             -------------------
    begin                : Oct 15 , 2015
    copyright            : (C) 2015 by Tomas Sobotik
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/


#include "MainBuildConfig.h"
	//for HOST versions:
#include "tnlFastSweepingMap.h"
	//for DEVICE versions:
//#include "tnlFastSweepingMap_CUDA.h"
#include "fastSweepingMapConfig.h"
#include <solvers/tnlBuildConfigTags.h>

#include <mesh/tnlGrid.h>
#include <core/tnlDevice.h>
#include <time.h>
#include <ctime>

typedef MainBuildConfig BuildConfig;

int main( int argc, char* argv[] )
{
	time_t start;
	time_t stop;
	time(&start);
	std::clock_t start2= std::clock();
   Config::ParameterContainer parameters;
   tnlConfigDescription configDescription;
   fastSweepingMapConfig< BuildConfig >::configSetup( configDescription );

   if( ! parseCommandLine( argc, argv, configDescription, parameters ) )
      return false;

   const int& dim = parameters.getParameter< int >( "dim" );

   if(dim == 2)
   {
		tnlFastSweepingMap<tnlGrid<2,double,TNL::Devices::Host, int>, double, int> solver;
		if(!solver.init(parameters))
	   {
			cerr << "Solver failed to initialize." <<std::endl;
			return EXIT_FAILURE;
	   }
		TNL_CHECK_CUDA_DEVICE;
	  std::cout << "-------------------------------------------------------------" <<std::endl;
	  std::cout << "Starting solver..." <<std::endl;
	   solver.run();
   }
//   else if(dim == 3)
//   {
//		tnlFastSweepingMap<tnlGrid<3,double,TNL::Devices::Host, int>, double, int> solver;
//		if(!solver.init(parameters))
//	   {
//			cerr << "Solver failed to initialize." <<std::endl;
//			return EXIT_FAILURE;
//	   }
//		TNL_CHECK_CUDA_DEVICE;
//	  std::cout << "-------------------------------------------------------------" <<std::endl;
//	  std::cout << "Starting solver..." <<std::endl;
//	   solver.run();
//   }
   else
   {
	  std::cerr << "Unsupported number of dimensions: " << dim << "!" <<std::endl;
	   return EXIT_FAILURE;
   }


   time(&stop);
  std::cout << "Solver stopped..." <<std::endl;
  std::cout <<std::endl;
  std::cout << "Running time was: " << difftime(stop,start) << " .... " << (std::clock() - start2) / (double)(CLOCKS_PER_SEC) <<std::endl;
   return EXIT_SUCCESS;
}


