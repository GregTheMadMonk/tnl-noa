/***************************************************************************
                          mcurve2gnuplot.cpp  -  description
                             -------------------
    begin                : 2007/12/16
    copyright            : (C) 2007 by Tomá¹ Oberhuber
    email                : oberhuber@seznam.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include <diff/mdiff.h>
#include "mcurve2gnuplot-def.h"

//--------------------------------------------------------------------------
int main( int argc, char* argv[] )
{
   mParameterContainer parameters;
   mConfigDescription conf_desc;
   if( conf_desc. ParseConfigDescription( CONFIG_DESCRIPTION_FILE ) != 0 )
      return 1;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc. PrintUsage( argv[ 0 ] );
      return 1;
   }

   mList< mString > input_files = parameters. GetParameter< mList< mString > >( "input-files" );
   mList< mString > output_files;
   if( ! parameters. GetParameter< mList< mString > >( "output-files", output_files ) )
      cout << "No output files were given." << endl;
   int output_step( 1 );
   parameters. GetParameter< int >( "output-step", output_step );
   mString output_file_format = parameters. GetParameter< mString >( "output-file-format" );

   long int size = input_files. Size();
   /*if( size != output_files. Size() )
   {
      cerr << "Sorry, there is different number of input and output files." << endl;
      return 1;
   }*/
   long int i;
   mCurve< mVector< 2, double > > crv;
   for( i = 0; i < size; i ++ )
   {
      const char* input_file = input_files[ i ]. Data();
      cout << "Processing file " << input_file << " ... " << flush;
      int strln = strlen( input_file );
      mString uncompressed_file_name( input_file );
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
      if( ! crv. Load( file ) )
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

      mCurve< mVector< 2, double > > out_crv;
      const long int size = crv. Size();
      long int i;
      for( i = 0; i < size; i += output_step )
      {
         out_crv. Append( crv[ i ]. position, crv[ i ]. separator );
         //mVector< 2, double > v = crv[ i ]. position;
         //v[ 0 ] = u( i );
         //v[ 1 ] = u( i + 1 );
         //out_crv. Append( v );
      }

      mString output_file_name;
      if( ! output_files. IsEmpty() ) output_file_name = output_files[ i ];
      else
      {
         if( strcmp( uncompressed_file_name. Data() + uncompressed_file_name. Length() - 4, ".bin" ) == 0 )
            output_file_name. SetString( uncompressed_file_name. Data(), 0, 4 );
         else output_file_name. SetString( uncompressed_file_name. Data() );
         if( output_file_format == "gnuplot" )
            output_file_name += ".gplt";
      }
      cout << " writing... " << output_file_name << endl;
      if( ! Write( out_crv, output_file_name. Data(), output_file_format. Data() ) )
      {
         cerr << " unable to write to " << output_file_name << endl;
      }
   }
}
