/***************************************************************************
                          tnlFile.cpp  -  description
                             -------------------
    begin                : Oct 22, 2010
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

#include <tnl-config.h>
#include <core/tnlFile.h>

int tnlFile :: verbose = 0;

tnlFile :: tnlFile()
: mode( tnlUndefinedMode ),
  compression( tnlCompressionNone ),
  file( NULL ),
  fileOK( false ),
  writtenElements( 0 ),
  readElements( 0 )
{
}

bool tnlFile :: open( const tnlString& fileName,
                      const tnlIOMode mode,
                      const tnlCompression compression )
{
   this -> fileName = fileName;
   this -> compression = compression;
   if( verbose )
   {
      cout << "Opening file " << fileName;
      if( mode == tnlReadMode )
         cout << " for reading... " << endl;
      else
         cout << " for writing ... " << endl;
   }
   switch( compression )
   {
      case tnlCompressionNone:
         break;
      case tnlCompressionBzip2:
#ifdef HAVE_BZIP2
         if( mode == tnlReadMode )
            file = fopen( fileName. getString(), "r" );
         if( mode == tnlWriteMode )
            file = fopen( fileName. getString(), "w" );
         if( file ==  NULL )
         {
            cerr << "I am not able to open the file " << fileName << ". ";
            perror( "" );
            return false;
         }
         int bzerror;
         if( mode == tnlReadMode)
            bzFile = BZ2_bzReadOpen( &bzerror,
                                     file,
                                     0, // int verbosity,
                                     0, // int small,
                                     NULL, // void *unused,
                                     0 //int nUnused
                                     );
         if( mode == tnlWriteMode )
            bzFile = BZ2_bzWriteOpen( &bzerror,
                                     file,
                                     9, //int blockSize100k,
                                     0, //int verbosity,
                                     0 //int workFactor
                                     );
         if( bzerror == BZ_OK )
         {
            fileOK = true;
            this -> mode = mode;
            return true;
         }
         if( bzerror == BZ_CONFIG_ERROR )
         {
            cerr << "ERROR: Cannot open the file " << fileName << " because the bzip2 library has been mis-compiled." << endl;
            return false;
         }
         if( bzerror == BZ_MEM_ERROR )
         {
            cerr << "ERROR: Cannot open the file " << fileName << " because of insufficient memory." << endl;
            return false;
         }
         cerr << "ERROR: Unknown error occured while I tried to open the file " << fileName << endl;
         checkBz2Error( bzerror );
         return false;
#else
         cerr << "Bzip2 compression is not supported on this system." << endl;
         return false;
#endif
         break;
      case tnlCompressionGzip:
         break;
   }
   return false;
}

bool tnlFile :: close()
{
   if( verbose )
      cout << "Closing the file " << getFileName() << " ... " << endl;
   switch( compression )
   {
      case tnlCompressionNone:
         break;
      case tnlCompressionBzip2:
#ifdef HAVE_BZIP2
         if( ! fileOK )
            return false;
         int bzerror;
         if( mode == tnlReadMode )
            BZ2_bzReadClose( &bzerror, bzFile );
         if( mode == tnlWriteMode )
            BZ2_bzWriteClose( &bzerror,
                              bzFile,
                              0, // int abandon,
                              0, // unsigned int* nbytes_in,
                              0  // unsigned int* nbytes_out
                              );
         mode = tnlUndefinedMode;
         if( bzerror != BZ_OK )
            cerr << "ERROR: I was not able to close the file " << fileName << endl;
         if( fclose( file ) != 0 )
         {
            cerr << "I was not able to close the file " << fileName << " properly!" << endl;
            return false;
         }
#else
         cerr << "Bzip2 compression is not supported on this system." << endl;
         return false;
#endif
         break;
      case tnlCompressionGzip:
         break;
   }
   readElements = writtenElements = 0;
   return true;
};

bool tnlFile :: checkBz2Error( int bzerror ) const
{
   switch( bzerror )
   {
      case BZ_OK:
         return true;
      case BZ_STREAM_END:
         return true;
      case BZ_CONFIG_ERROR:
         cerr << "The BZ2 library has been mis-compiled" << endl;
         return false;
      case BZ_PARAM_ERROR:
         cerr << "It is a libbz2 parameter error. It must be a bug in the program." << endl;
         return false;
      case BZ_IO_ERROR:
         cerr << "It is a I/O error." << endl;
         return false;
      case BZ_MEM_ERROR:
         cerr << "Insufficient memory for I/O operation." << endl;
         return false;
      case BZ_UNEXPECTED_EOF:
         cerr << "Unexpected end of file." << endl;
         return false;
      default:
         cerr << "Unknown error with code " << bzerror << endl;
   }
};


