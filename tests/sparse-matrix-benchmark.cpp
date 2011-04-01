/***************************************************************************
                          sparse-matrix-benchmark.cpp  -  description
                             -------------------
    begin                : Jul 27, 2010
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

#include "sparse-matrix-benchmark-def.h"

#include "sparse-matrix-benchmark.h"
#include <string.h>
#include <core/tnlObject.h>
#include <core/tnlFile.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;

   if( conf_desc. ParseConfigDescription( CONFIG_DESCRIPTION_FILE ) != 0 )
      return 1;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc. PrintUsage( argv[ 0 ] );
      return 1;
   }
   tnlString str_input_file = parameters. GetParameter< tnlString >( "input-file" );
   tnlString str_input_mtx_file = parameters. GetParameter< tnlString >( "input-mtx-file" );
   tnlString log_file_name = parameters. GetParameter< tnlString >( "log-file" );
   double stop_time = parameters. GetParameter< double >( "stop-time" );
   int verbose = parameters. GetParameter< int >( "verbose");
   char* input_file = str_input_file. getString();

   tnlFile binaryFile;
   if( ! binaryFile. open( str_input_file, tnlReadMode ) )
   {
      cerr << "I am not able to open the file " << str_input_file << "." << endl;
      return 1;
   }
   tnlString object_type;
   if( ! getObjectType( binaryFile, object_type ) )
   {
      cerr << "Unknown object ... SKIPPING!" << endl;
      return EXIT_FAILURE;
   }
   if( verbose )
      cout << object_type << " detected ... " << endl;
   binaryFile. close();

   spmvBenchmarkStatistics benchmarkStatistics;

   int size, nonzero_elements;

   if( object_type == "tnlCSRMatrix< float, tnlHost >")
      benchmarkMatrix< float >( str_input_file,
               str_input_mtx_file,
               verbose,
               stop_time,
               size,
               nonzero_elements,
               benchmarkStatistics );

   if( object_type == "tnlCSRMatrix< double, tnlHost >" )
   {
#if CUDA_ARCH > 12
      benchmarkMatrix< double >( str_input_file,
               str_input_mtx_file,
               verbose,
               stop_time,
               size,
               nonzero_elements,
               benchmarkStatistics );
#else
   cerr << "Skipping double precision test because this CUDA device does not support the double precision." << endl;
#endif
   }
   //binaryFile. close();


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
      double all_elements = ( double ) size;
      benchmarkStatistics. writeToLog( log_file,
                                       str_input_file,
                                       size,
                                       nonzero_elements,
                                       all_elements * all_elements );
   }
   return EXIT_SUCCESS;
}
