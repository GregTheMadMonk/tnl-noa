/***************************************************************************
                          bin2gnuplot.cpp  -  description
                             -------------------
    begin                : 2007/06/28
    copyright            : (C) 2007 by Tomas Oberhuber
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

#include <fstream>
#include <mdiff.h>
#include <stdio.h>

//--------------------------------------------------------------------------
int main( int argc, char* argv[] )
{
   int i;
   mGrid2D< double > u;
   for( i = 1; i < argc; i ++ )
   {
      const char* file_name = argv[ i ];
      cout << "Processing file " << file_name << " ... " << flush;
      int strln = strlen( file_name );
      mString uncompressed_file_name;
      if( strcmp( file_name + strln - 3, ".gz" ) == 0 )
         if( ! UnCompressFile( file_name, "gz" ) )
         {
            cerr << "Unable to uncompress the file " << file_name << "." << endl;
            return -1;
         }
         else uncompressed_file_name. SetString( file_name, 0, 3 );
      if( strcmp( file_name + strln - 4, ".bz2" ) == 0 )
         if( ! UnCompressFile( file_name, "bz2" ) )
         {
            cerr << "Unable to uncompress the file " << file_name << "." << endl;
            return -1;
         }
         else uncompressed_file_name. SetString( file_name, 0, 4 );

          
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
      if( strcmp( file_name + strln - 3, ".gz" ) == 0 &&
          ! CompressFile( uncompressed_file_name. Data(), "gz" ) )
      {
         cerr << "Unable to compress back the file " << file_name << "." << endl;
         return -1;
      }
      if( strcmp( file_name + strln - 4, ".bz2" ) == 0 &&
          ! CompressFile( uncompressed_file_name. Data(), "bz2" ) )
      {
         cerr << "Unable to compress back the file " << file_name << "." << endl;
         return -1;
      }

      mString output_file_name;
      output_file_name. SetString( uncompressed_file_name. Data(), 0, 4 );
      cout << " writing... " << output_file_name;
      if( ! Draw( u, output_file_name. Data(), "gnuplot" ) )
      {
         cout << " unable to write to " << output_file_name << endl;
      }
      cout << " ... OK " << endl;
   }

}
