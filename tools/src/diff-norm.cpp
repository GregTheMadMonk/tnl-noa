/***************************************************************************
                          diff-norm.cpp  -  description
                             -------------------
    begin                : 2007/07/05
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

#include <fstream>
#include <diff/tnlGrid2D.h>
#include <stdio.h>

bool ReadInputFile( const char* file_name, 
                    tnlGrid2D< double >& u )
{
   cout << "I am processing the file..." << file_name << " ... " << flush;
   int strln = strlen( file_name );
   tnlString uncompressed_file_name( file_name );
   if( strcmp( file_name + strln - 3, ".gz" ) == 0 )
      if( ! UnCompressFile( file_name, "gz" ) )
      {
         cerr << "Sorry, I am unable to uncompress the file " << file_name << "." << endl;
         return false;
      }
      else uncompressed_file_name. SetString( file_name, 0, 3 );
   if( strcmp( file_name + strln - 4, ".bz2" ) == 0  )
      if( ! UnCompressFile( file_name, "bz2" ) )
      {
         cerr << "Sorry, I am unable to uncompress the file " << file_name << "." << endl;
         return false;
      }
      else uncompressed_file_name. SetString( file_name, 0, 4 );
       
   fstream file;
   file. open( uncompressed_file_name. Data(), ios :: in | ios :: binary );
   if( ! file )
   {
      cout << "Sorry, I cannot open the file " << uncompressed_file_name << endl;
      return false;
   }
   if( ! u. Load( file ) )
   {
      cout << "Sorry, I cannot restore the data. " << endl;
      return false;
   }
   file. close();
   if( strcmp( file_name + strln - 3, ".gz" ) == 0 &&
       ! CompressFile( uncompressed_file_name. Data(), "gz" ) )
   {
      cerr << "Sorry, I am not able to compress the file " << file_name << " back." << endl;
      return false;
   }
   if( strcmp( file_name + strln - 4, ".bz2" ) == 0 &&
       ! CompressFile( uncompressed_file_name. Data(), "bz2" ) )
   {
      cerr << "Sorry, I am not able to compress the file " << file_name << " back." << endl;
      return false;
   }
   return true;
}
//--------------------------------------------------------------------------
int main( int argc, char* argv[] )
{
   int i;
   tnlGrid2D< double > u1, u2;
   bool have_u2( false );
   if( argc == 1 )
   {
      cerr << "I am missing input file." << endl;
      return -1;
   }
   if( ! ReadInputFile( argv[ 1 ], u1 ) )
      return -1;
   cout << endl;
   if( argc >= 2 && ReadInputFile( argv[ 2 ], u2 ) )
      have_u2 = true;
   cout << endl;

   if( ! have_u2 )
   {
      cout << "Having only one input file I am going to compute its norms:" << endl;
      cout << "L1 norm is: " << GetL1Norm( u1 ) << endl;
      cout << "L2 norm is: " << GetL2Norm( u1 ) << endl;
      cout << "Max. norm is: " << GetMaxNorm( u1 ) << endl;
   }
   else
   {
      cout << "I have two input files so I am going to compute the norms of the difference:" << endl;
      cout << "L1 norm is: " << GetDiffL1Norm( u1, u2 ) << endl;
      cout << "L2 norm is: " << GetDiffL2Norm( u1, u2 ) << endl;
      cout << "Max. norm is: " << GetDiffMaxNorm( u1, u2 ) << endl;
   }
   cout << "Bye." << endl;
}

