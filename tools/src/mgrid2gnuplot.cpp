/***************************************************************************
                          mgrid2gnuplot.cpp  -  description
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

#include "mgrid2gnuplot-def.h"

#include <diff/mGrid2D.h>
#include <core/mParameterContainer.h>


//--------------------------------------------------------------------------
int main( int argc, char* argv[] )
{
   mParameterContainer parameters;
   tnlConfigDescription conf_desc;
   if( conf_desc. ParseConfigDescription( CONFIG_DESCRIPTION_FILE ) != 0 )
      return 1;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc. PrintUsage( argv[ 0 ] );
      return 1;
   }

   tnlList< tnlString > input_files = parameters. GetParameter< tnlList< tnlString > >( "input-files" );
   tnlList< tnlString > output_files;
   if( ! parameters. GetParameter< tnlList< tnlString > >( "output-files", output_files ) )
      cout << "No output files were given." << endl;
   tnlList< double > level_lines;
   if( ! parameters. GetParameter< tnlList< double > >( "level-lines", level_lines ) )
      cout << "No level lines were given." << endl;
   int output_x_size( 0 ), output_y_size( 0 );
   parameters. GetParameter< int >( "output-x-size", output_x_size );
   parameters. GetParameter< int >( "output-y-size", output_y_size );
   double scale = parameters. GetParameter< double >( "scale" );
   tnlString output_file_format = parameters. GetParameter< tnlString >( "output-file-format" );

   long int size = input_files. Size();
   /*if( size != output_files. Size() )
   {
      cerr << "Sorry, there is different number of input and output files." << endl;
      return 1;
   }*/
   long int i;
   mGrid2D< double > u;
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
            return -1;
         }
         else uncompressed_file_name. SetString( input_file, 0, 3 );
      if( strcmp( input_file + strln - 4, ".bz2" ) == 0 )
         if( ! UnCompressFile( input_file, "bz2" ) )
         {
            cerr << "Unable to uncompress the file " << input_file << "." << endl;
            return -1;
         }
         else uncompressed_file_name. SetString( input_file, 0, 4 );

          
      fstream file;
      file. open( uncompressed_file_name. Data(), ios :: in | ios :: binary );
      if( ! file )
      {
         cout << " unable to open file " << uncompressed_file_name. Data() << endl;
         continue;
      }
      if( ! u. Load( file ) )
      {
         cout << " unable to restore the data " << endl;
         continue;
      }
      file. close();
      if( strcmp( input_file + strln - 3, ".gz" ) == 0 &&
          ! CompressFile( uncompressed_file_name. Data(), "gz" ) )
      {
         cerr << "Unable to compress back the file " << input_file << "." << endl;
         return -1;
      }
      if( strcmp( input_file + strln - 4, ".bz2" ) == 0 &&
          ! CompressFile( uncompressed_file_name. Data(), "bz2" ) )
      {
         cerr << "Unable to compress back the file " << input_file << "." << endl;
         return -1;
      }

      mGrid2D< double >* output_u;
      if( ! output_x_size && ! output_y_size && scale == 1.0 )
         output_u = &u;
      else
      {
         if( ! output_x_size ) output_x_size = u. GetXSize();
         if( ! output_y_size ) output_y_size = u. GetYSize();

         output_u = new mGrid2D< double >( output_x_size,
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

      tnlString output_file_name;
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
      }
      cout << " writing ... " << output_file_name;
      if( ! level_lines. IsEmpty() )
      {
         tnlCurve< mVector< 2, double > > crv;
         long int j;
         for( j = 0; j < level_lines. Size(); j ++ )
            if( ! GetLevelSetCurve( * output_u, crv, level_lines[ j ] ) )
            {
               cerr << "Unable to identify the level line " << level_lines[ j ] << endl;
               return 1;
            }
         if( ! Write( crv, output_file_name. Data(), output_file_format. Data() ) )
         {
            cerr << " unable to write to " << output_file_name << endl;
         }
      }
      else
         if( ! Draw( *output_u, output_file_name. Data(), output_file_format. Data() ) )
         {
            cerr << " unable to write to " << output_file_name << endl;
         }
      cout << " ... OK " << endl;
      if( output_u != &u ) delete output_u;
   }
}
