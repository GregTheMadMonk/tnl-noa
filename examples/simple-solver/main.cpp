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
#include "simpleProblemTypesSetter.h"
#include <config/tnlBasicTypesSetter.h>
#include <solvers/tnlProblemSolver.h>

int main( int argc, char* argv[] )
{
   typedef simpleProblemTypesSetter ProblemSetter;
   typedef tnlBasicTypesSetter< ProblemSetter > BasicTypesSetter;
   tnlProblemSolver< BasicTypesSetter > problemSolver;
   if( ! problemSolver. run( CONFIG_FILE, argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}


