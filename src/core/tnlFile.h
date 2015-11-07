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
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_CUDA
   #include <cuda_runtime.h>
#endif

#include <core/mfuncs.h>
#include <core/tnlAssert.h>
#include <core/tnlString.h>
#include <core/tnlHost.h>
#include <core/tnlCuda.h>

using namespace std;

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

   FILE* file;

   bool fileOK;

   tnlString fileName;

   long int writtenElements;

   long int readElements;

   public:

   tnlFile();

   bool open( const tnlString& fileName,
              const tnlIOMode mode );


	const tnlString& getFileName() const
   {
	   return this->fileName;
   }

	long int getReadElements() const
	{
	   return this->readElements;
	}

	long int getWrittenElements() const
	{
	   return this->writtenElements;
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
	bool write( const Type* buffer );
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
   bool write( const Type* buffer );

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
bool tnlFile :: write( const Type* buffer )
{
   return write< Type, Device, int >( buffer, 1 );
};


template< typename Type, typename Device, typename Index >
bool tnlFile :: read( Type* buffer,
                      const Index& elements )
{
   tnlAssert( elements >= 0,
              cerr << " elements = " << elements << endl; );
   if( ! elements )
      return true;
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
   this->readElements = 0;
   const Index host_buffer_size = :: Min( ( Index ) ( tnlFileGPUvsCPUTransferBufferSize / sizeof( Type ) ),
                                          elements );
   void* host_buffer( 0 );
   if( Device :: getDeviceType() == "tnlHost" )
   {
      if( fread( buffer,
             sizeof( Type ),
             elements,
             file ) != elements )
      {
         cerr << "I am not able to read the data from the file " << fileName << "." << endl;
         perror( "Fread ended with the error code" );
         return false;
      }
      this->readElements = elements;
      return true;
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
         size_t transfered = fread( host_buffer, sizeof( Type ), transfer, file );
         if( transfered != transfer )
         {
            cerr << "I am not able to read the data from the file " << fileName << "." << endl;
            cerr << transfered << " bytes were transfered. " << endl;
            perror( "Fread ended with the error code" );
            return false;
         }

         cudaMemcpy( ( void* ) & ( buffer[ readElements ] ),
                     host_buffer,
                     transfer * sizeof( Type ),
                     cudaMemcpyHostToDevice );
         if( ! checkCudaDevice )
         {
            cerr << "Transfer of data from the CUDA device to the file " << this -> fileName
                 << " failed." << endl;
            free( host_buffer );
            return false;
         }
         readElements += transfer;
      }
      free( host_buffer );
      return true;
#else
      tnlCudaSupportMissingMessage;;
      return false;
#endif
   }
   return true;
};

template< class Type, typename Device, typename Index >
bool tnlFile ::  write( const Type* buffer,
                        const Index elements )
{
   tnlAssert( elements >= 0,
              cerr << " elements = " << elements << endl; );
   if( ! elements )
      return true;
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

   Type* buf = const_cast< Type* >( buffer );
   void* host_buffer( 0 );
   this->writtenElements = 0;
   const Index host_buffer_size = :: Min( ( Index ) ( tnlFileGPUvsCPUTransferBufferSize / sizeof( Type ) ),
                                          elements );
   if( Device :: getDeviceType() == "tnlHost" )
   {
      if( fwrite( buf,
                  sizeof( Type ),
                  elements,
                  this->file ) != elements )
      {
         cerr << "I am not able to write the data to the file " << fileName << "." << endl;
         perror( "Fwrite ended with the error code" );
         return false;
      }
      this->writtenElements = elements;
      return true;
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

         while( this->writtenElements < elements )
         {
            Index transfer = :: Min( elements - this->writtenElements, host_buffer_size );
            cudaMemcpy( host_buffer,
                       ( void* ) & ( buffer[ this->writtenElements ] ),
                       transfer * sizeof( Type ),
                       cudaMemcpyDeviceToHost );
            if( ! checkCudaDevice )
            {
               cerr << "Transfer of data from the file " << this -> fileName
                    << " to the CUDA device failed." << endl;
               free( host_buffer );
               return false;
            }
            if( fwrite( host_buffer,
                        sizeof( Type ),
                        transfer,
                        this->file ) != transfer )
            {
               cerr << "I am not able to write the data to the file " << fileName << "." << endl;
               perror( "Fwrite ended with the error code" );
               return false;
            }
            this->writtenElements += transfer;
         }
         free( host_buffer );
         return true;
#else
         tnlCudaSupportMissingMessage;;
         return false;
#endif
   }
   return true;
};

inline bool fileExists( const tnlString& fileName )
{
  fstream file;
  file.open( fileName. getString(), ios::in );
  bool result( true );
  if( ! file )
     result = false;
  file.close();
  return result;
};

#endif /* TNLFILE_H_ */
