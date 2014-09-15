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

#include "heatEquationSetter.h"
#include <solvers/tnlSolver.h>
#include <solvers/tnlFastBuildConfig.h>
#include <solvers/tnlConfigTags.h>

//typedef tnlDefaultConfigTag BuildConfig;
typedef tnlFastBuildConfig BuildConfig;

template< typename ConfigTag >
class heatEquationConfig
{
   public:
      static void configSetup( tnlConfigDescription& config )
      {
         config.addDelimiter( "Heat equation settings:" );
         config.addDelimiter( "Tests setting::" );
      }
};

int main( int argc, char* argv[] )
{
   tnlSolver< heatEquationSetter, heatEquationConfig, BuildConfig > solver;
   if( ! solver. run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}


