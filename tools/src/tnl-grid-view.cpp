/***************************************************************************
                          mgrid-view.cpp  -  description
                             -------------------
    begin                : 2007/08/20
    copyright            : (C) 2007 by Tomas Oberhuber
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

#include "tnl-grid-view.h"
#include <core/tnlCurve.h>
#include <core/tnlFile.h>
#include <debug/tnlDebug.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <diff/curve-ident.h>

#include "configDirectory.h"
const char configFile[] = CONFIG_DIRECTORY "tnl-grid-view.cfg.desc";

int main( int argc, char* argv[] )
{
   dbgFunctionName( "", "main" );
   dbgInit( "" );
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;
   if( conf_desc. ParseConfigDescription( configFile ) != 0 )
      return 1;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc. PrintUsage( argv[ 0 ] );
      return 1;
   }

   tnlList< tnlString > input_files = parameters. GetParameter< tnlList< tnlString > >( "input-files" );
   int verbose = parameters. GetParameter< int >( "verbose");

   int size = input_files. getSize();
   tnlString output_file_name;
   tnlList< tnlString > output_files;
   if( ! parameters. GetParameter< tnlList< tnlString > >( "output-files", output_files ) )
      cout << "No output files were given." << endl;
   int i;
   for( i = 0; i < size; i ++ )
   {
      const char* input_file = input_files[ i ]. getString();
      if( verbose )
         cout << "Processing file " << input_file << " ... " << flush;

       
       tnlString object_type;
       if( ! getObjectType( input_files[ i ]. getString(), object_type ) )
          cerr << "unknown object ... SKIPPING!" << endl;
       else
       {
          if( verbose )
             cout << object_type << " detected ... ";

         tnlString output_file_format = parameters. GetParameter< tnlString >( "output-format" );
         if( ! output_files. isEmpty() ) output_file_name = output_files[ i ];
         else
         {
            if( strcmp( input_file + input_files[ i ]. getLength() - 4, ".tnl" ) == 0 )
               output_file_name. setString( input_file, 0, 4 );
            else output_file_name. setString( input_file );
            output_file_name += ".crv";
            if( output_file_format == "tnl" )
               output_file_name += ".tnl";
            if( output_file_format == "gnuplot" )
               output_file_name += ".gplt";
            if( output_file_format == "vti" )
               output_file_name += ".vti";
            if( output_file_format == "povray" )
               output_file_name += ".df3";
         }

         bool object_type_matched( false );
         if( object_type == "tnlGrid< 2, float, tnlHost, int >" )
         {
            ProcesstnlGrid2D< float, tnlHost, int >( input_file, parameters, i, output_file_name, output_file_format );
            object_type_matched = true;
         }
         if( object_type == "tnlGrid< 2, double, tnlHost, int >" )
         {
            ProcesstnlGrid2D< double, tnlHost, int >( input_file, parameters, i, output_file_name, output_file_format );
            object_type_matched = true;
         }
         if( object_type == "tnlGrid< 3, float, tnlHost, int >" )
         {
            ProcesstnlGrid3D< float, tnlHost, int >( input_file, parameters, i, output_file_name, output_file_format );
            object_type_matched = true;
         }
         if( object_type == "tnlGrid< 3, double, tnlHost, int >" )
         {
             ProcesstnlGrid3D< double, tnlHost, int >( input_file, parameters, i, output_file_name, output_file_format );
             object_type_matched = true;
         }
         if( object_type == "tnlGrid< 2, tnlFloat, tnlHost, int >" )
         {
            ProcesstnlGrid2D< tnlFloat, tnlHost, int >( input_file, parameters, i, output_file_name, output_file_format );
            object_type_matched = true;
         }
         if( object_type == "tnlGrid< 2, tnlDouble, tnlHost, int >" )
         {
            ProcesstnlGrid2D< tnlDouble, tnlHost, int >( input_file, parameters, i, output_file_name, output_file_format );
            object_type_matched = true;
         }
         if( object_type == "tnlGrid< 3, tnlFloat, tnlHost, int >" )
         {
            ProcesstnlGrid3D< tnlFloat, tnlHost, int >( input_file, parameters, i, output_file_name, output_file_format );
            object_type_matched = true;
         }
         if( object_type == "tnlGrid3D< tnlDouble >" )
         {
             ProcesstnlGrid3D< tnlDouble, tnlHost, int >( input_file, parameters, i, output_file_name, output_file_format );
             object_type_matched = true;
         }
         if( object_type == "tnlCSRMatrix< float, tnlHost >")
         {
            ProcessCSRMatrix< float >( input_file, parameters, i, output_file_name, output_file_format );
            object_type_matched = true;
         }
         if( object_type == "tnlCSRMatrix< double, tnlHost >")
         {
            ProcessCSRMatrix< double >( input_file, parameters, i, output_file_name, output_file_format );
            object_type_matched = true;
         }
         if( ! object_type_matched )
            cerr << "Unknown type ... Skipping. " << endl;
       }
   
   }
   return 0;
}
