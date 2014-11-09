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

#include "hamiltonJacobiProblemConfig.h"
#include "hamiltonJacobiProblemSetter.h"
#include <solvers/tnlSolver.h>
#include "MainBuildConfig.h"
#include <solvers/tnlConfigTags.h>

typedef MainBuildConfig BuildConfig;

int main( int argc, char* argv[] )
{
   tnlSolver< hamiltonJacobiProblemSetter, hamiltonJacobiProblemConfig, BuildConfig > solver;
   if( ! solver. run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}


