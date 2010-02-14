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

#include "mgrid-view-def.h"
#include "tnl-grid-view.h"
#include <core/compress-file.h>
#include <core/tnlCurve.h>
#include <core/tnlConfigDescription.h>
#include <core/tnlParameterContainer.h>
#include <diff/curve-ident.h>
#include <diff/drawGrid2D.h>
#include <diff/drawGrid3D.h>

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;
   if( conf_desc. ParseConfigDescription( MGRID_VIEW_CONFIG_DESCRIPTION_FILE ) != 0 )
      return 1;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc. PrintUsage( argv[ 0 ] );
      return 1;
   }

   tnlList< tnlString > input_files = parameters. GetParameter< tnlList< tnlString > >( "input-files" );
   int verbose = parameters. GetParameter< int >( "verbose");

   int size = input_files. Size();
   tnlString output_file_name;
   tnlList< tnlString > output_files;
   if( ! parameters. GetParameter< tnlList< tnlString > >( "output-files", output_files ) )
      cout << "No output files were given." << endl;
   int i;
   for( i = 0; i < size; i ++ )
   {
      const char* input_file = input_files[ i ]. Data();
      if( verbose )
         cout << "Processing file " << input_file << " ... " << flush;

      int strln = strlen( input_file );
      tnlString uncompressed_file_name( input_file );
      if( strcmp( input_file + strln - 3, ".gz" ) == 0 )
         if( ! UnCompressFile( input_file, "gz" ) )
         {
            cerr << "Unable to uncompress the file " << input_file << "." << endl;
            return false;
         }
         else uncompressed_file_name. SetString( input_file, 0, 3 );
      if( strcmp( input_file + strln - 4, ".bz2" ) == 0 )
         if( ! UnCompressFile( input_file, "bz2" ) )
         {
            cerr << "Unable to uncompress the file " << input_file << "." << endl;
            return false;
         }
         else uncompressed_file_name. SetString( input_file, 0, 4 );

       
       tnlString object_type;
       if( ! GetObjectType( uncompressed_file_name. Data(), object_type ) )
          cerr << "unknown object ... SKIPPING!" << endl;
       else
       {
          if( verbose )
             cout << object_type << " detected ... ";

         tnlString output_file_format = parameters. GetParameter< tnlString >( "output-file-format" );
         if( ! output_files. IsEmpty() ) output_file_name = output_files[ i ];
         else
         {
            if( strcmp( uncompressed_file_name. Data() + uncompressed_file_name. Length() - 4, ".bin" ) == 0 )
               output_file_name. SetString( uncompressed_file_name. Data(), 0, 4 );
            else output_file_name. SetString( uncompressed_file_name. Data() );
            if( output_file_format == "gnuplot" )
               output_file_name += ".gplt";
            if( output_file_format == "vti" )
               output_file_name += ".vti";
            if( output_file_format == "povray" )
               output_file_name += ".df3";
         }

         bool object_type_matched( false );
         if( object_type == "tnlGrid2D< float >" )
         {
            ProcesstnlGrid2D< float >( uncompressed_file_name. Data(), parameters, i, output_file_name, output_file_format );
            object_type_matched = true;
         }
         if( object_type == "tnlGrid2D< double >" )
         {
            ProcesstnlGrid2D< double >( uncompressed_file_name. Data(), parameters, i, output_file_name, output_file_format );
            object_type_matched = true;
         }
         if( object_type == "tnlGrid3D< float >" )
         {
            ProcesstnlGrid3D< float >( uncompressed_file_name. Data(), parameters, i, output_file_name, output_file_format );
            object_type_matched = true;
         }
         if( object_type == "tnlGrid3D< double >" )
         {
             ProcesstnlGrid3D< double >( uncompressed_file_name. Data(), parameters, i, output_file_name, output_file_format );
             object_type_matched = true;
         }
         if( ! object_type_matched )
            cerr << "Unknown type ... Skipping. " << endl;
       }
   
       if( strcmp( input_file + strln - 3, ".gz" ) == 0 &&
           ! CompressFile( uncompressed_file_name. Data(), "gz" ) )
       {
          cerr << "Unable to compress back the file " << input_file << "." << endl;
          return false;
       }
       if( strcmp( input_file + strln - 4, ".bz2" ) == 0 &&
           ! CompressFile( uncompressed_file_name. Data(), "bz2" ) )
       {
          cerr << "Unable to compress back the file " << input_file << "." << endl;
          return false;
       }
   }
   return 0;
}
