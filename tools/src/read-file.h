/***************************************************************************
                          read-file.h  -  description
                             -------------------
    begin                : 2008/02/13
    copyright            : (C) 2008 by Tomas Oberhuber
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

#ifndef read_fileH
#define read_fileH

#include "src/core/mcore.h"
#include "src/diff/mdiff.h"

template< class T > bool ReadFile( const char* input_file, T& u )
{
   int strln = strlen( input_file );
   mString uncompressed_file_name( input_file );
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

       
   fstream file;
   file. open( uncompressed_file_name. Data(), ios :: in | ios :: binary );
   if( ! file )
   {
      cout << " unable to open file " << uncompressed_file_name. Data() << endl;
      return false;
   }
   if( ! u. Load( file ) )
   {
      cerr << " unable to restore the data " << endl;
      return false;
   }
   file. close();
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
   return true;
}

#endif
