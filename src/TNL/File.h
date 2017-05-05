/***************************************************************************
                          File.h  -  description
                             -------------------
    begin                : 8 Oct 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_CUDA
   #include <cuda_runtime.h>
#endif

#include <TNL/Assert.h>
#include <TNL/String.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {

enum tnlIOMode { tnlUndefinedMode = 0,
                 tnlReadMode = 1,
                 tnlWriteMode = 2 };

/* When we need to transfer data between the GPU and the CPU we use
 * 5 MB buffer. This size should ensure good performance -- see.
 * http://wiki.accelereyes.com/wiki/index.php/GPU_Memory_Transfer
 * Similar constant is defined in tnlLonegVectorCUDA
 */
const size_t tnlFileGPUvsCPUTransferBufferSize = 5 * 2<<20;

/*
 * This file is aimed mainly for the binary data. It supports transparent compression.
 */
class File
{
   tnlIOMode mode;

   FILE* file;

   bool fileOK;

   String fileName;

   size_t writtenElements;

   size_t readElements;

   public:

   File();

   ~File();

   bool open( const String& fileName,
              const tnlIOMode mode );


	const String& getFileName() const
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
   template< typename Type, typename Device = Devices::Host, typename Index = int >
   bool read( Type* buffer,
              const Index& elements );

   template< typename Type, typename Device = Devices::Host >
   bool read( Type* buffer );

   template< typename Type, typename Device = Devices::Host, typename Index = int >
   bool write( const Type* buffer,
               const Index elements );

   template< typename Type, typename Device = Devices::Host >
   bool write( const Type* buffer );

#endif

	bool close();

	static int verbose;

};

bool fileExists( const String& fileName );

} // namespace TNL

#include <TNL/File_impl.h>
