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

#include "heat-equation-conf.h"
#include "heatEquationSetter.h"
#include <solvers/tnlSolver.h>

int main( int argc, char* argv[] )
{
   tnlSolver< heatEquationSetter > solver;
   if( ! solver. run( CONFIG_FILE, argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}


