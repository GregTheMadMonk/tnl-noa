/***************************************************************************
                          matrix-formats-test.cpp  -  description
                             -------------------
    begin                : Dec 14, 2013
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

#include "matrix-formats-test.h"
#include <cstdlib>
#include <core/tnlFile.h>
#include <debug/tnlDebug.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>

#include "tnlConfig.h"
const char configFile[] = TNL_CONFIG_DIRECTORY "tnl-test-matrix-formats.cfg.desc";

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;
   if( conf_desc. ParseConfigDescription( configFile ) != 0 )
      return EXIT_FAILURE;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc. PrintUsage( argv[ 0 ] );
      return EXIT_FAILURE;
   }

   const tnlString& matrixFormat = parameters.GetParameter< tnlString >( "matrix-format" );
   if( matrixFormat == "dense" )
   {
       if( !testMatrix< tnlDenseMatrix >() )
          return EXIT_FAILURE;
       return EXIT_SUCCESS:
   }
   if( matrixFormat == "ellpack" )
   {
       if( !testMatrix< tnlEllpackMatrix >() )
          return EXIT_FAILURE;
       return EXIT_SUCCESS:
   }
   if( matrixFormat == "sliced-ellpack" )
   {
       if( !testMatrix< tnlSlicedEllpackMatrix >() )
          return EXIT_FAILURE;
       return EXIT_SUCCESS:
   }
   if( matrixFormat == "chunked-ellpack" )
   {
       if( !testMatrix< tnlChunkedEllpackMatrix >() )
          return EXIT_FAILURE;
       return EXIT_SUCCESS:
   }
   if( matrixFormat == "csr" )
   {
       if( !testMatrix< tnlCSRMatrix >() )
          return EXIT_FAILURE;
       return EXIT_SUCCESS:
   }
   cerr << "Uknown matrix format " << matrixFormat << "." << endl;
   return EXIT_FAILURE;
}


