/***************************************************************************
                          tnl-matrix-convert.h  -  description
                             -------------------
    begin                : Jul 23, 2010
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

#ifndef TNLMATRIXCONVERT_H_
#define TNLMATRIXCONVERT_H_

#include <fstream>
#include <core/tnlString.h>
#include <core/tnlFile.h>
#include <matrix/tnlCSRMatrix.h>

using namespace std;

template< class REAL >
bool convertMatrix( const tnlString& input_file,
                    const tnlString& output_file,
                    const tnlString& output_matrix_format,
                    int verbose,
                    bool verify )
{
   tnlMatrix< REAL >* matrix( NULL ), *verify_matrix( NULL );
   if( output_matrix_format == "csr" )
   {
      matrix = new tnlCSRMatrix< REAL, tnlHost >( input_file. getString() );
      if( verify )
         verify_matrix = new tnlCSRMatrix< REAL, tnlHost >( "verify-matrix" );
   }
   if( ! matrix )
   {
      cerr << "I was not able to create matrix with format " << output_matrix_format << "." << endl;
      if( verify_matrix ) delete verify_matrix;
      return false;
   }
   if( verify && ! verify_matrix )
   {
      cerr << "I was not able to create the verification matrix with format " << output_matrix_format << "." << endl;
      if( matrix ) delete matrix;
      return false;
   }

   fstream file;
   file. open( input_file. getString(), ios :: in );
   if( ! file )
   {
      cerr << "I was not able to open the input file " << input_file << "." << endl;
      return false;
   }
   if( ! matrix -> read( file, verbose ) )
   {
      cerr << "I was not able to read the matrix from the file " << input_file << "." << endl;
      file. close();
      return false;
   }
   file. close();

   if( verbose )
      cout << endl << "Writing to a file " << output_file << " ... " << flush;
   tnlFile binaryFile;
   if( ! binaryFile. open( output_file. getString(), tnlWriteMode, tnlCompressionBzip2 ) )
   {
      cerr << endl << "I was not able to open the output file " << output_file << "." << endl;
      return false;
   }
   if( ! matrix -> save( binaryFile ) )
   {
      cerr << endl << "I was not able to write binary data of the matrix format "
           << output_matrix_format << " to the file " << output_file << endl;
      file. close();
      return false;
   }
   binaryFile. close();
   if( verbose )
	   cout << "OK." << endl;

   if( verify_matrix )
   {
      if( verbose )
         cout << "Verifying the output matrix ... " << flush;
      if( ! binaryFile. open( output_file. getString(), tnlReadMode ) )
      {
         cerr << "Unable to open the file " << output_file << " for the verification." << endl;
         delete matrix;
         delete verify_matrix;
         return false;
      }
      if( ! verify_matrix -> load( binaryFile ) )
      {
         cerr << "Unable to restore the verification matrix." << endl;
         delete matrix;
         delete verify_matrix;
         return false;
      }
      if( ( *matrix ) != ( *verify_matrix ) )
      {
         cerr << "The matrix verification failed !!!" << endl;
         delete matrix;
         delete verify_matrix;
         return false;
      }
      if( verbose )
         cout << "OK." << endl;
      delete verify_matrix;
   }
   /*if( verbose )
            cout << "Compressing output matrix ... ";
   if( ! CompressFile( output_file. getString(), "bz2" ) )
   {
      cerr << "Unable to compress the output file " << output_file << "." << endl;
      delete matrix;
      return false;
   }
   if( verbose )
      cout << "OK." << endl;*/
   delete matrix;
   return true;
}
#endif /* TNLMATRIXCONVERT_H_ */
