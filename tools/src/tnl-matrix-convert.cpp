/***************************************************************************
                          tnl-matrix-convert.cpp  -  description
                             -------------------
    begin                : Jul 23, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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

#include "tnl-matrix-convert.h"
#include <string.h>
#include <debug/tnlDebug.h>
#include <core/tnlObject.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>

#include "tnlConfig.h"
const char configFile[] = TNL_CONFIG_DIRECTORY "tnl-matrix-convert.cfg.desc";

int main( int argc, char* argv[] )
{
   dbgFunctionName( "", "main" );
   dbgInit( "" );

   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;

   if( conf_desc.parseConfigDescription( configFile ) != 0 )
      return 1;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc.printUsage( argv[ 0 ] );
      return 1;
   }
   tnlString input_file = parameters. GetParameter< tnlString >( "input-file" );
   tnlString output_file = parameters. GetParameter< tnlString >( "output-file" );
   tnlString output_matrix_format = parameters. GetParameter< tnlString >( "output-matrix-format" );
   tnlString precision = parameters. GetParameter< tnlString >( "precision" );
   int verbose = parameters. GetParameter< int >( "verbose");
   bool verify = parameters. GetParameter< bool >( "verify");

   if( verbose )
      cout << "Processing file " << input_file << " ... " << endl;

   if( precision == "float" )
   {
      if( ! convertMatrix< float >( input_file,
                                    output_file,
                                    output_matrix_format,
                                    verbose,
                                    verify ) )
         return EXIT_FAILURE;
      else
         return EXIT_SUCCESS;
   }

   if( precision == "double" )
   {
      if( ! convertMatrix< double >( input_file,
                                     output_file,
                                     output_matrix_format,
                                     verbose,
                                     verify ) )
         return EXIT_FAILURE;
      else
         return EXIT_SUCCESS;
   }


   cerr << "Unknnown precision " << precision << " was given. Can be only float of double." << endl;
   return EXIT_FAILURE;
}
