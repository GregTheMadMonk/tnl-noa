/***************************************************************************
                          File.h  -  description
                             -------------------
    begin                : 8 Oct 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <fstream>
#include <type_traits>

#include <TNL/String.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/MIC.h>

namespace TNL {

/**
 * When we transfer data between the GPU and the CPU we use 5 MB buffer. This
 * size should ensure good performance -- see.
 * http://wiki.accelereyes.com/wiki/index.php/GPU_Memory_Transfer
 */
static constexpr std::streamsize FileGPUvsCPUTransferBufferSize = 5 * 2<<20;

/**
 * \brief This class serves for binary IO. It allows to do IO even for data allocated on GPU
 *
 * \par Example
 * \include FileExample.cpp
 * \par Output
 * \include FileExample.out
 */
class File
{
   public:

      /**
       * This enum defines mode for opening files.
       */
      enum class Mode
      {
         In = 1,       ///< Open for input.
         Out = 2,      ///< Open for output.
         Append = 4,   ///< Output operations are appended at the end of file.
         AtEnd = 8,    ///< Set the initial position at the end.
         Truncate = 16 ///< If the file is opened for ouptput, its previous content is deleted.
      };
      
      /**
       * \brief Basic constructor.
       */
      File() = default;

      /**
       * \brief Open given file and returns \e true after the file is
       * successfully opened. Otherwise returns \e false.
       *
       * Opens file with given \e fileName and returns true/false based on the success in opening the file.
       * \param fileName String which indicates name of the file user wants to open.
       * \param mode Indicates what user needs to do with opened file.
       */
      bool open( const String& fileName,
                 Mode mode = static_cast< Mode >( static_cast< int >( Mode::In ) | static_cast< int >( Mode::Out ) ) );

      /**
       * \brief Attempts to close given file and returns \e true when the file is
       * successfully closed. Otherwise returns \e false.
       */
      bool close();

      /**
       * \brief Returns name of given file.
       */
      const String& getFileName() const
      {
         return this->fileName;
      }

      /**
       * \brief Method that can write particular data type from given file into GPU. (Function that gets particular elements from given file.)
       *
       * Returns \e true when the elements are successfully read from given file. Otherwise returns \e false.
       *
       * Throws \ref std::ios_base::failure on failure.
       *
       * \tparam Type Type of data.
       * \tparam Device Place where data are stored after reading from file. For example \ref Devices::Host or \ref Devices::Cuda.
       * \tparam Index Type of index by which the elements are indexed.
       * \param buffer Pointer in memory where the elements are loaded and stored after reading.
       * \param elements Number of elements the user wants to get (read) from given file.
       */
      template< typename Type, typename Device = Devices::Host >
      bool read( Type* buffer, std::streamsize elements = 1 );

      /**
       * \brief Method that can write particular data type from CPU into given file. (Function that writes particular elements into given file.)
       *
       * Returns \e true when the elements are successfully written into given file. Otherwise returns \e false.
       *
       * Throws \ref std::ios_base::failure on failure.
       *
       * \tparam Type Type of data.
       * \tparam Device Place from where the data are loaded before writing into file. For example \ref Devices::Host or \ref Devices::Cuda.
       * \tparam Index Type of index by which the elements are indexed.
       * \param buffer Pointer in memory where the elements are loaded from before writing into file.
       * \param elements Number of elements the user wants to write into the given file.
       */
      template< typename Type, typename Device = Devices::Host >
      bool write( const Type* buffer, std::streamsize elements = 1 );

   protected:
      template< typename Type,
                typename Device,
                typename = typename std::enable_if< std::is_same< Device, Devices::Host >::value >::type >
      bool read_impl( Type* buffer, std::streamsize elements );

      template< typename Type,
                typename Device,
                typename = typename std::enable_if< std::is_same< Device, Devices::Cuda >::value >::type,
                typename = void >
      bool read_impl( Type* buffer, std::streamsize elements );

      template< typename Type,
                typename Device,
                typename = typename std::enable_if< std::is_same< Device, Devices::MIC >::value >::type,
                typename = void,
                typename = void >
      bool read_impl( Type* buffer, std::streamsize elements );

      template< typename Type,
                typename Device,
                typename = typename std::enable_if< std::is_same< Device, Devices::Host >::value >::type >
      bool write_impl( const Type* buffer, std::streamsize elements );

      template< typename Type,
                typename Device,
                typename = typename std::enable_if< std::is_same< Device, Devices::Cuda >::value >::type,
                typename = void >
      bool write_impl( const Type* buffer, std::streamsize elements );

      template< typename Type,
                typename Device,
                typename = typename std::enable_if< std::is_same< Device, Devices::MIC >::value >::type,
                typename = void,
                typename = void >
      bool write_impl( const Type* buffer, std::streamsize elements );

      std::fstream file;
      String fileName;
};

/**
 * \brief Returns true if the file exists and false otherwise.
 *
 * Finds out if the file \e fileName exists.
 * \param fileName Name of the file to check.
 */
bool fileExists( const String& fileName );

/**
 * \brief Serialization of strings
 */
File& operator<<( File& file, const std::string& str );

/**
 * \brief Deserialization of strings.
 */
File& operator>>( File& file, std::string& str );
} // namespace TNL

#include <TNL/File.hpp>
