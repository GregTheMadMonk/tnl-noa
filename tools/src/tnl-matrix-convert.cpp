/***************************************************************************
                          tnl-matrix-convert.cpp  -  description
                             -------------------
    begin                : Jul 23, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include "tnl-matrix-convert.h"
#include <string.h>
#include <TNL/debug/tnlDebug.h>
#include <TNL/Object.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>

#include "tnlConfig.h"
const char configFile[] = TNL_CONFIG_DIRECTORY "tnl-matrix-convert.cfg.desc";

int main( int argc, char* argv[] )
{
   dbgFunctionName( "", "main" );
   dbgInit( "" );

   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   if( conf_desc.parseConfigDescription( configFile ) != 0 )
      return 1;
   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc.printUsage( argv[ 0 ] );
      return 1;
   }
   String input_file = parameters. getParameter< String >( "input-file" );
   String output_file = parameters. getParameter< String >( "output-file" );
   String output_matrix_format = parameters. getParameter< String >( "output-matrix-format" );
   String precision = parameters. getParameter< String >( "precision" );
   int verbose = parameters. getParameter< int >( "verbose");
   bool verify = parameters. getParameter< bool >( "verify");

   if( verbose )
      std::cout << "Processing file " << input_file << " ... " << std::endl;

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


   std::cerr << "Unknnown precision " << precision << " was given. Can be only float of double." << std::endl;
   return EXIT_FAILURE;
}
