/***************************************************************************
                          main.h  -  description
                             -------------------
    begin                : Mar 22 , 2016
    copyright            : (C) 2016 by Tomas Sobotik
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "tnlParallelMapSolver.h"
#include "parallelMapConfig.h"
#include "MainBuildConfig.h"
#include <solvers/tnlBuildConfigTags.h>
#include <operators/hamilton-jacobi/godunov-eikonal/parallelGodunovMap.h>
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
	parallelMapConfig< BuildConfig >::configSetup( configDescription );

	if( ! parseCommandLine( argc, argv, configDescription, parameters ) )
	  return false;


	tnlDeviceEnum device;
	device = TNL::Devices::HostDevice;

	const int& dim = parameters.getParameter< int >( "dim" );

	if(dim == 2)
	{

	   typedef parallelGodunovMapScheme< tnlGrid<2,double,TNL::Devices::Host, int>, double, int > SchemeTypeHost;
/*#ifdef HAVE_CUDA
		   typedef parallelGodunovMapScheme< tnlGrid<2,double,tnlCuda, int>, double, int > SchemeTypeDevice;
#endif
#ifndef HAVE_CUDA*/
	   typedef parallelGodunovMapScheme< tnlGrid<2,double,TNL::Devices::Host, int>, double, int > SchemeTypeDevice;
/*#endif*/

	   if(device==TNL::Devices::HostDevice)
	   {
		   typedef TNL::Devices::Host Device;


		   tnlParallelMapSolver<2,SchemeTypeHost,SchemeTypeDevice, Device> solver;
		   if(!solver.init(parameters))
		   {
			  std::cerr << "Solver failed to initialize." <<std::endl;
			   return EXIT_FAILURE;
		   }
		  std::cout << "-------------------------------------------------------------" <<std::endl;
		  std::cout << "Starting solver loop..." <<std::endl;
		   solver.run();
	   }
	   else if(device==tnlCudaDevice )
	   {
		   typedef tnlCuda Device;
//typedef parallelGodunovMapScheme< tnlGrid<2,double,Device, int>, double, int > SchemeType;

		   tnlParallelMapSolver<2,SchemeTypeHost,SchemeTypeDevice, Device> solver;
		   if(!solver.init(parameters))
		   {
			  std::cerr << "Solver failed to initialize." <<std::endl;
			   return EXIT_FAILURE;
		   }
		  std::cout << "-------------------------------------------------------------" <<std::endl;
		  std::cout << "Starting solver loop..." <<std::endl;
		   solver.run();
	   }
	}


	time(&stop);
	cout <<std::endl;
	cout << "Running time was: " << difftime(stop,start) << " .... " << (std::clock() - start2) / (double)(CLOCKS_PER_SEC) <<std::endl;
	return EXIT_SUCCESS;
}


