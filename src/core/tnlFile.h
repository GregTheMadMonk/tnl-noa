/***************************************************************************
                          tnlFile.h  -  description
                             -------------------
    begin                : 8 Oct 2010
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

#ifndef TNLFILE_H_
#define TNLFILE_H_

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <tnlConfig.h>
#ifdef HAVE_BZIP2
   #include <bzlib.h>
#endif
#ifdef HAVE_CUDA
   #include <cuda_runtime.h>
#endif

#include <core/mfuncs.h>
#include <core/tnlAssert.h>
#include <legacy/core/tnlCudaSupport.h>
#include <core/tnlString.h>
#include <core/tnlObject.h>
#include <core/tnlHost.h>
#include <core/tnlCuda.h>

using namespace std;

enum tnlCompression { tnlCompressionNone = 0,
                      tnlCompressionBzip2,
                      tnlCompressionGzip };

enum tnlIOMode { tnlUndefinedMode = 0,
                 tnlReadMode = 1,
                 tnlWriteMode = 2 };

/* When we need to transfer data between the GPU and the CPU we use
 * 5 MB buffer. This size should ensure good performance -- see.
 * http://wiki.accelereyes.com/wiki/index.php/GPU_Memory_Transfer
 * Similar constant is defined in tnlLonegVectorCUDA
 */
const int tnlFileGPUvsCPUTransferBufferSize = 5 * 2<<20;


/*
 * This file is aimed mainly for the binary data. It supports transparent compression.
 */
class tnlFile
{
   tnlIOMode mode;

   tnlCompression compression;

   FILE* file;

   bool fileOK;

   tnlString fileName;

   long int writtenElements;

   long int readElements;

#ifdef HAVE_BZIP2
   BZFILE* bzFile;
#endif

   bool checkBz2Error( int bzerror ) const;

   public:

   tnlFile();

	bool open( const tnlString& fileName,
                   const tnlIOMode mode,
		   const tnlCompression compression = tnlCompressionBzip2 );

	const tnlString& getFileName() const
   {
	   return fileName;
   }

	long int getReadElements() const
	{
	   return readElements;
	}

	long int getWrittenElements() const
	{
	   return writtenElements;
	}

	// TODO: this does not work for constant types
#ifdef HAVE_NOT_CXX11
	template< typename Type, typename Device, typename Index >
	bool read( Type* buffer,
	           const Index& elements );

	template< typename Type, typename Device >
	bool read( Type* buffer );

	template< typename Type, typename Device, typename Index >
	bool write( const Type* buffer,
	            const Index elements );

	template< typename Type, typename Device >
	bool write( Type* buffer );
#else        
   template< typename Type, typename Device = tnlHost, typename Index = int >
   bool read( Type* buffer,
              const Index& elements );

   template< typename Type, typename Device = tnlHost >
   bool read( Type* buffer );

   template< typename Type, typename Device = tnlHost, typename Index = int >
   bool write( const Type* buffer,
               const Index elements );

   template< typename Type, typename Device = tnlHost >
   bool write( Type* buffer );

#endif

	bool close();

	static int verbose;

};

template< typename Type, typename Device >
bool tnlFile :: read( Type* buffer )
{
   return read< Type, Device, int >( buffer, 1 );
};

template< typename Type, typename Device >
bool tnlFile :: write( Type* buffer )
{
   return write< Type, Device, int >( buffer, 1 );
};


template< typename Type, typename Device, typename Index >
bool tnlFile :: read( Type* buffer,
                         const Index& elements )
{
   if( ! fileOK )
   {
      cerr << "File " << fileName << " was not properly opened. " << endl;
      return false;
   }
   if( mode != tnlReadMode )
   {
      cerr << "File " << fileName << " was not opened for reading. " << endl;
      return false;
   }
#ifdef HAVE_BZIP2
   int bzerror;
   int bytes_read( 0 );
   const Index host_buffer_size = :: Min( ( Index ) ( tnlFileGPUvsCPUTransferBufferSize / sizeof( Type ) ),
                                          elements );
   void* host_buffer( 0 );
   if( Device :: getDeviceType() == "tnlHost" )
   {
         bytes_read = BZ2_bzRead( &bzerror,
                                  bzFile,
                                  buffer,
                                  elements * sizeof( Type ) );
         if( bzerror == BZ_OK || bzerror == BZ_STREAM_END )
            readElements = bytes_read / sizeof( Type );
         return checkBz2Error( bzerror );
   }
   if( Device :: getDeviceType() == "tnlCuda" )
   {
#ifdef HAVE_CUDA
      /*!***
       * Here we cannot use
       *
       * host_buffer = new Type[ host_buffer_size ];
       *
       * because it does not work for constant types like
       * T = const bool.
       */
      host_buffer = malloc( sizeof( Type ) * host_buffer_size );
      readElements = 0;
      if( ! host_buffer )
      {
         cerr << "I am sorry but I cannot allocate supporting buffer on the host for writing data from the GPU to the file "
              << this -> getFileName() << "." << endl;
         return false;

      }

      while( readElements < elements )
      {
         int transfer = :: Min( ( Index ) ( elements - readElements ), host_buffer_size );
         int bytesRead = BZ2_bzRead( &bzerror,
                                     bzFile,
                                     host_buffer,
                                     transfer * sizeof( Type ) );
         if( ! checkBz2Error( bzerror) )
         {
            free( host_buffer );
            return false;
         }
         if( cudaMemcpy( ( void* ) & ( buffer[ readElements ] ),
                         host_buffer,
                         bytesRead,
                         cudaMemcpyHostToDevice ) != cudaSuccess )
         {
            cerr << "Transfer of data from the CUDA device to the file " << this -> fileName
                 << " failed." << endl;
            checkCUDAError( __FILE__, __LINE__ );
            free( host_buffer );
            return false;
         }
         readElements += bytesRead / sizeof( Type );
      }
      free( host_buffer );
      return true;
#else
      cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
      return false;
#endif
   }
#else
   cerr << "Bzip2 compression is not supported on this system." << endl;
   return false;
#endif
   return true;
};

template< class Type, typename Device, typename Index >
bool tnlFile ::  write( const Type* buffer,
                           const Index elements )
{
   if( ! fileOK )
   {
      cerr << "File " << fileName << " was not properly opened. " << endl;
      return false;
   }
   if( mode != tnlWriteMode )
   {
      cerr << "File " << fileName << " was not opened for writing. " << endl;
      return false;
   }
#ifdef HAVE_BZIP2
   int bzerror;
   Type* buf = const_cast< Type* >( buffer );
   void* host_buffer( 0 );
   Index writtenElements( 0 );
   const Index host_buffer_size = :: Min( ( Index ) ( tnlFileGPUvsCPUTransferBufferSize / sizeof( Type ) ),
                                          elements );
   if( Device :: getDeviceType() == "tnlHost" )
   {
      BZ2_bzWrite( &bzerror,
                   bzFile,
                   ( void* ) buf,
                   elements * sizeof( Type ) );
      return checkBz2Error( bzerror );
   }
   if( Device :: getDeviceType() == "tnlCuda" )
   {
#ifdef HAVE_CUDA
         /*!***
          * Here we cannot use
          *
          * host_buffer = new Type[ host_buffer_size ];
          *
          * because it does not work for constant types like
          * T = const bool.
          */
         host_buffer = malloc( sizeof( Type ) * host_buffer_size );
         if( ! host_buffer )
         {
            cerr << "I am sorry but I cannot allocate supporting buffer on the host for writing data from the GPU to the file "
                 << this -> getFileName() << "." << endl;
            return false;
         }

         while( writtenElements < elements )
         {
            Index transfer = :: Min( elements - writtenElements, host_buffer_size );
            if( cudaMemcpy( host_buffer,
                            ( void* ) & ( buffer[ writtenElements ] ),
                            transfer * sizeof( Type ),
                            cudaMemcpyDeviceToHost ) != cudaSuccess )
            {
               checkCUDAError( __FILE__, __LINE__ );
               cerr << "CUDA error." << endl;
               free( host_buffer );
               return false;
            }
            BZ2_bzWrite( &bzerror,
                         bzFile,
                         ( void* ) host_buffer,
                         transfer * sizeof( Type ) );
            if( ! checkBz2Error( bzerror ) )
            {
               free( host_buffer );
               return false;
            }
            writtenElements += transfer;
         }
         free( host_buffer );
         return true;
#else
         cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
         return false;
#endif
   }

#else
   cerr << "Bzip2 compression is not supported on this system." << endl;
   return false;
#endif
   return true;
};


#endif /* TNLFILE_H_ */
