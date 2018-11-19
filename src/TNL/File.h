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
const size_t FileGPUvsCPUTransferBufferSize = 5 * 2<<20;


/// Class file is aimed mainly for the binary data.
///
/// \par Example
/// \include FileExample.cpp
// \par Output
// \include FileExample.out
class File
{
   IOMode mode;

   std::FILE* file;

   bool fileOK;

   String fileName;

   std::size_t writtenElements;

   std::size_t readElements;

   public:

   /// Basic constructor.
   File();

   /// Destructor.
   ~File();

   /////
   /// \brief Opens given file.
   ///
   /// Opens file with given \e fileName and returns true/false based on the success in opening the file.
   /// \param fileName String which indicates name of the file user wants to open.
   /// \param mode Indicates what user needs to do with opened file.
   /// Modes to choose: IOMode::read or IOMode::write or IOMode::undefined.
   bool open( const String& fileName,
              const IOMode mode );

   /// \brief Returns name of given file.
   const String& getFileName() const
   {
      return this->fileName;
   }

   /// Returns number of read elements.
   long int getReadElements() const
   {
      return this->readElements;
   }

   /// Returns number of written elements.
   long int getWrittenElements() const
   {
      return this->writtenElements;
   }

   /// \brief Method that can write particular data type from given file into GPU. (Function that gets particular elements from given file.)
   ///
   /// Returns boolean value based on the succes in reading elements from given file.
   /// \param buffer Pointer in memory (where the read elements are stored?).
   /// \param elements Number of elements the user wants to get (read) from given file.
   template< typename Type, typename Device = Devices::Host, typename Index = int >
   bool read( Type* buffer,
              const Index& elements );

   // Toto je treba??
   template< typename Type, typename Device = Devices::Host >
   bool read( Type* buffer );

   /// \brief Method that can write particular data type from CPU into given file. (Function that writes particular elements into given file.)
   ///
   /// Returns boolean value based on the succes in writing elements into given file.
   /// \param buffer Pointer in memory.
   /// \param elements Number of elements the user wants to write into the given file.
   template< typename Type, typename Device = Devices::Host, typename Index = int >
   bool write( const Type* buffer,
               const Index elements );

   // Toto je treba?
   template< typename Type, typename Device = Devices::Host >
   bool write( const Type* buffer );

   /// \brief Closes given file and returns true/false based on the success in closing the file.
   bool close();

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

/// Returns true if the file exists and false otherwise.
///
/// Finds out if the file \e fileName exists.
/// \param fileName Name of the file that user wants to find in the PC.
bool fileExists( const String& fileName );

} // namespace TNL

#include <TNL/File_impl.h>
