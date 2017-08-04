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
#include <cstdio>

#include <TNL/Assert.h>
#include <TNL/String.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/MIC.h>

namespace TNL {

enum class IOMode
{
   undefined = 0,
   read = 1,
   write = 2
};

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
   IOMode mode;

   std::FILE* file;

   bool fileOK;

   String fileName;

   std::size_t writtenElements;

   std::size_t readElements;

   public:

   File();

   ~File();

   bool open( const String& fileName,
              const IOMode mode );


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

   bool close();

   static int verbose;

protected:
   template< typename Type,
             typename Device,
             typename = typename std::enable_if< std::is_same< Device, Devices::Host >::value >::type >
   bool read_impl( Type* buffer,
                   const std::size_t& elements );

   template< typename Type,
             typename Device,
             typename = typename std::enable_if< std::is_same< Device, Devices::Cuda >::value >::type,
             typename = void >
   bool read_impl( Type* buffer,
                   const std::size_t& elements );

   template< typename Type,
             typename Device,
             typename = typename std::enable_if< std::is_same< Device, Devices::MIC >::value >::type,
             typename = void,
             typename = void >
   bool read_impl( Type* buffer,
                   const std::size_t& elements );

   template< typename Type,
             typename Device,
             typename = typename std::enable_if< std::is_same< Device, Devices::Host >::value >::type >
   bool write_impl( const Type* buffer,
                    const std::size_t& elements );

   template< typename Type,
             typename Device,
             typename = typename std::enable_if< std::is_same< Device, Devices::Cuda >::value >::type,
             typename = void >
   bool write_impl( const Type* buffer,
                    const std::size_t& elements );

   template< typename Type,
             typename Device,
             typename = typename std::enable_if< std::is_same< Device, Devices::MIC >::value >::type,
             typename = void,
             typename = void >
   bool write_impl( const Type* buffer,
                    const std::size_t& elements );
};

bool fileExists( const String& fileName );

} // namespace TNL

#include <TNL/File_impl.h>
