/***************************************************************************
                          mgrid-view.cpp  -  description
                             -------------------
    begin                : 2007/08/20
    copyright            : (C) 2007 by Tomá¹ Oberhuber
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

#include <core/compress-file.h>
#include <core/tnlCurve.h>
#include <core/tnlConfigDescription.h>
#include <core/tnlParameterContainer.h>
#include <diff/curve-ident.h>
#include <diff/drawGrid2D.h>
#include <diff/drawGrid3D.h>

bool ProcesstnlGrid2D( const tnlString& file_name, 
                     const tnlParameterContainer& parameters,
                     int file_index,
                     const tnlString& output_file_name,
                     const tnlString& output_file_format )
{
   tnlGrid2D< double > u;
   fstream file;
   file. open( file_name. Data(), ios :: in | ios :: binary );
   if( ! u. Load( file ) )
   {
      cout << " unable to restore the data " << endl;
      file. close();
      return false;
   }
   file. close();

   tnlGrid2D< double >* output_u;
   
   int output_x_size( 0 ), output_y_size( 0 );
   parameters. GetParameter< int >( "output-x-size", output_x_size );
   parameters. GetParameter< int >( "output-y-size", output_y_size );
   double scale = parameters. GetParameter< double >( "scale" );
   if( ! output_x_size && ! output_y_size && scale == 1.0 )
      output_u = &u;
   else
   {
      if( ! output_x_size ) output_x_size = u. GetXSize();
      if( ! output_y_size ) output_y_size = u. GetYSize();

      output_u = new tnlGrid2D< double >( output_x_size,
                                  output_y_size,
                                  u. GetAx(),
                                  u. GetBx(),
                                  u. GetAy(),
                                  u. GetBy() );
      const double& hx = output_u -> GetHx();
      const double& hy = output_u -> GetHy();
      long int i, j;
      for( i = 0; i < output_x_size; i ++ )
         for( j = 0; j < output_y_size; j ++ )
         {
            const double x = output_u -> GetAx() + i * hx;
            const double y = output_u -> GetAy() + j * hy;
            ( *output_u )( i, j ) = scale * u. Value( x, y );
         }
   }

   cout << " writing ... " << output_file_name;

   tnlList< double > level_lines;
   parameters. GetParameter< tnlList< double > >( "level-lines", level_lines );
   if( ! level_lines. IsEmpty() )
   {
      tnlCurve< tnlVector< 2, double > > crv;
      long int j;
      for( j = 0; j < level_lines. Size(); j ++ )
         if( ! GetLevelSetCurve( * output_u, crv, level_lines[ j ] ) )
         {
            cerr << "Unable to identify the level line " << level_lines[ j ] << endl;
            if( output_u != &u ) delete output_u;
            return false;
         }
      if( ! Write( crv, output_file_name. Data(), output_file_format. Data() ) )
      {
         cerr << " ... FAILED " << endl;
      }
   }
   else
      if( ! Draw( *output_u, output_file_name. Data(), output_file_format. Data() ) )
      {
         cerr << " ... FAILED " << endl;
      }
   if( output_u != &u ) delete output_u;
   cout << " OK " << endl;
}
//--------------------------------------------------------------------------
bool ProcesstnlGrid3D( const tnlString& file_name,
                     const tnlParameterContainer& parameters,
                     int file_index,
                     const tnlString& output_file_name,
                     const tnlString& output_file_format )
{
   tnlGrid3D< double > u;
   fstream file;
   file. open( file_name. Data(), ios :: in | ios :: binary );
   if( ! u. Load( file ) )
   {
      cout << " unable to restore the data " << endl;
      file. close();
      return false;
   }
   file. close();

   tnlGrid3D< double >* output_u;
   
   int output_x_size( 0 ), output_y_size( 0 ), output_z_size( 0 );
   parameters. GetParameter< int >( "output-x-size", output_x_size );
   parameters. GetParameter< int >( "output-y-size", output_y_size );
   parameters. GetParameter< int >( "output-y-size", output_z_size );
   double scale = parameters. GetParameter< double >( "scale" );
   if( ! output_x_size && ! output_y_size && ! output_z_size && scale == 1.0 )
      output_u = &u;
   else
   {
      if( ! output_x_size ) output_x_size = u. GetXSize();
      if( ! output_y_size ) output_y_size = u. GetYSize();
      if( ! output_z_size ) output_z_size = u. GetZSize();

      output_u = new tnlGrid3D< double >( output_x_size,
                                        output_y_size,
                                        output_z_size,
                                        u. GetAx(),
                                        u. GetBx(),
                                        u. GetAy(),
                                        u. GetBy(),
                                        u. GetAz(),
                                        u. GetBz() );
      const double& hx = output_u -> GetHx();
      const double& hy = output_u -> GetHy();
      const double& hz = output_u -> GetHz();
      long int i, j, k;
      for( i = 0; i < output_x_size; i ++ )
         for( j = 0; j < output_y_size; j ++ )
            for( k = 0; j < output_y_size; k ++ )
            {
               const double x = output_u -> GetAx() + i * hx;
               const double y = output_u -> GetAy() + j * hy;
               const double z = output_u -> GetAz() + k * hz;
               ( *output_u )( i, j, k ) = scale * u. Value( x, y, z );
            }
   }

   cout << " writing " << output_file_name << " ... ";
   if( ! Draw( *output_u, output_file_name. Data(), output_file_format. Data() ) )
   {
      cerr << " unable to write to " << output_file_name << endl;
   }
   else
      cout << " ... OK " << endl;
   if( output_u != &u ) delete output_u;
}
//--------------------------------------------------------------------------
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

   long int size = input_files. Size();
   tnlString output_file_name;
   tnlList< tnlString > output_files;
   if( ! parameters. GetParameter< tnlList< tnlString > >( "output-files", output_files ) )
      cout << "No output files were given." << endl;
   long int i;
   for( i = 0; i < size; i ++ )
   {
      const char* input_file = input_files[ i ]. Data();
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
         if( object_type == "tnlGrid2D< double >" )
         {
            ProcesstnlGrid2D( uncompressed_file_name. Data(), parameters, i, output_file_name, output_file_format );
            object_type_matched = true;
         }
         if( object_type == "tnlGrid3D< double >" )
         {
             ProcesstnlGrid3D( uncompressed_file_name. Data(), parameters, i, output_file_name, output_file_format );
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
