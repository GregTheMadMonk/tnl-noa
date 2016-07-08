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

bool fileExists( const tnlString& fileName );


#include <core/tnlFile_impl.h>

#endif /* TNLFILE_H_ */
