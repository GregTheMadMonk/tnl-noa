/***************************************************************************
                          matrix-solvers-benchmark.cpp  -  description
                             -------------------
    begin                : Jan 8, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
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

#include "matrix-solvers-benchmark-def.h"

#include "matrix-solvers-benchmark.h"
#include <fstream>
#include <core/tnlObject.h>
#include <core/tnlFile.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>


int main( int argc, char* argv[] )
{
   /****
    * Parsing command line arguments ...
    */
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;

   if( conf_desc. ParseConfigDescription( CONFIG_DESCRIPTION_FILE ) != 0 )
      return 1;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc. PrintUsage( argv[ 0 ] );
      return 1;
   }
   tnlString inputFile = parameters. GetParameter< tnlString >( "input-file" );
   tnlString str_input_mtx_file = parameters. GetParameter< tnlString >( "input-mtx-file" );
   tnlString log_file_name = parameters. GetParameter< tnlString >( "log-file" );
   double stop_time = parameters. GetParameter< double >( "stop-time" );
   int verbose = parameters. GetParameter< int >( "verbose");

   /****
    * Checking a type of the input data
    */
   tnlString objectType;
   if( ! getObjectType( inputFile, objectType ) )
   {
      cerr << "Unable to detect object type in " << inputFile << endl;
      return EXIT_FAILURE;
   }
   tnlList< tnlString > parsedObjectType;
   parseObjectType( objectType,
                    parsedObjectType );
   tnlString objectClass = parsedObjectType[ 0 ];
   if( objectClass != "tnlCSRMatrix" )
   {
      cerr << "I am sorry, I am expecting tnlCSRMatrix in the input file but I found " << objectClass << "." << endl;
      return EXIT_FAILURE;
   }
   tnlString precision = parsedObjectType[ 1 ];
   //tnlString indexing = parsedObjectType[ 3 ];
   if( precision == "float" )
      if( ! benchmarkMatrix< float, int >( inputFile ) )
         return EXIT_FAILURE;
   if( precision == "double" )
      if( ! benchmarkMatrix< double, int >( inputFile ) )
         return EXIT_FAILURE;



   fstream log_file;
   if( log_file_name )
   {
      log_file. open( log_file_name. getString(), ios :: out | ios :: app );
      if( ! log_file )
      {
         cerr << "Unable to open log file " << log_file_name << " for appending logs." << endl;
         return EXIT_FAILURE;
      }
      cout << "Writing to log file " << log_file_name << "..." << endl;
   }
   return EXIT_SUCCESS;

}
