/***************************************************************************
                          tnl-matrix-convert.h  -  description
                             -------------------
    begin                : Jul 23, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLMATRIXCONVERT_H_
#define TNLMATRIXCONVERT_H_

#include <fstream>
#include <TNL/String.h>
#include <TNL/File.h>
#include <TNL/matrices/tnlMatrix.h>
#include <TNL/matrices/tnlCSRMatrix.h>

using namespace std;

template< class REAL >
bool convertMatrix( const String& input_file,
                    const String& output_file,
                    const String& output_matrix_format,
                    int verbose,
                    bool verify )
{
   /*tnlMatrix< REAL >* matrix( NULL ), *verify_matrix( NULL );
   if( output_matrix_format == "csr" )
   {
      matrix = new tnlCSRMatrix< REAL, tnlHost >( input_file. getString() );
      if( verify )
         verify_matrix = new tnlCSRMatrix< REAL, tnlHost >( "verify-matrix" );
   }
   if( ! matrix )
   {
      std::cerr << "I was not able to create matrix with format " << output_matrix_format << "." << std::endl;
      if( verify_matrix ) delete verify_matrix;
      return false;
   }
   if( verify && ! verify_matrix )
   {
      std::cerr << "I was not able to create the verification matrix with format " << output_matrix_format << "." << std::endl;
      if( matrix ) delete matrix;
      return false;
   }

   std::fstream file;
   file. open( input_file. getString(), std::ios::in );
   if( ! file )
   {
      std::cerr << "I was not able to open the input file " << input_file << "." << std::endl;
      return false;
   }
   if( ! matrix -> read( file, verbose ) )
   {
      std::cerr << "I was not able to read the matrix from the file " << input_file << "." << std::endl;
      file. close();
      return false;
   }
   file. close();

   if( verbose )
     std::cout << std::endl << "Writing to a file " << output_file << " ... " << std::flush;
   File binaryFile;
   if( ! binaryFile. open( output_file. getString(), tnlWriteMode ) )
   {
      std::cerr << std::endl << "I was not able to open the output file " << output_file << "." << std::endl;
      return false;
   }
   if( ! matrix -> save( binaryFile ) )
   {
      std::cerr << std::endl << "I was not able to write binary data of the matrix format "
           << output_matrix_format << " to the file " << output_file << std::endl;
      file. close();
      return false;
   }
   binaryFile. close();
   if( verbose )
	  std::cout << "OK." << std::endl;

   if( verify_matrix )
   {
      if( verbose )
        std::cout << "Verifying the output matrix ... " << std::flush;
      if( ! binaryFile. open( output_file. getString(), tnlReadMode ) )
      {
         std::cerr << "Unable to open the file " << output_file << " for the verification." << std::endl;
         delete matrix;
         delete verify_matrix;
         return false;
      }
      if( ! verify_matrix -> load( binaryFile ) )
      {
         std::cerr << "Unable to restore the verification matrix." << std::endl;
         delete matrix;
         delete verify_matrix;
         return false;
      }
      if( ( *matrix ) != ( *verify_matrix ) )
      {
         std::cerr << "The matrix verification failed !!!" << std::endl;
         delete matrix;
         delete verify_matrix;
         return false;
      }
      if( verbose )
        std::cout << "OK." << std::endl;
      delete verify_matrix;
   }*/
   /*if( verbose )
           std::cout << "Compressing output matrix ... ";
   if( ! CompressFile( output_file. getString(), "bz2" ) )
   {
      std::cerr << "Unable to compress the output file " << output_file << "." << std::endl;
      delete matrix;
      return false;
   }
   if( verbose )
     std::cout << "OK." << std::endl;*/
   //delete matrix;
   return true;
}
#endif /* TNLMATRIXCONVERT_H_ */
