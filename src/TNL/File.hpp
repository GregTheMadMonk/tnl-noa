/***************************************************************************
                          File_impl.h  -  description
                             -------------------
    begin                : Mar 5, Oct 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <memory>
#include <iostream>

#include <TNL/File.h>
#include <TNL/Assert.h>
#include <TNL/Exceptions/CudaSupportMissing.h>
#include <TNL/Exceptions/MICSupportMissing.h>
#include <TNL/Exceptions/FileSerializationError.h>
#include <TNL/Exceptions/FileDeserializationError.h>

namespace TNL {

inline bool File::open( const String& fileName, const IOMode mode )
{
   // enable exceptions
   file.exceptions( std::fstream::failbit | std::fstream::badbit | std::fstream::eofbit );

   close();

   if( mode == IOMode::read )
      file.open( fileName.getString(), std::ios::binary | std::ios::in );
   else
      file.open( fileName.getString(), std::ios::binary | std::ios::out );

   this->fileName = fileName;
   if( ! file.good() ) {
      std::cerr << "I am not able to open the file " << fileName << ". ";
      return false;
   }
   return true;
}

inline bool File::close()
{
   if( file.is_open() ) {
      file.close();
      if( ! file.good() ) {
         std::cerr << "I was not able to close the file " << fileName << " properly!" << std::endl;
         return false;
      }
   }
   // reset all attributes
   fileName = "";
   return true;
}

/*template< typename Type, typename Device >
bool File::read( Type* buffer )
{
   return read< Type, Device >( buffer, 1 );
}*/

/*template< typename Type, typename Device >
bool File::write( const Type* buffer )
{
   return write< Type, Device >( buffer, 1 );
}*/

template< typename Type, typename Device >
bool File::read( Type* buffer, std::streamsize elements )
{
   TNL_ASSERT_GE( elements, 0, "Number of elements to read must be non-negative." );

   if( ! elements )
      return true;

   return read_impl< Type, Device >( buffer, elements );
}

// Host
template< typename Type,
          typename Device,
          typename >
bool File::read_impl( Type* buffer, std::streamsize elements )
{
   file.read( reinterpret_cast<char*>(buffer), sizeof(Type) * elements );
   return true;
}

// Cuda
template< typename Type,
          typename Device,
          typename, typename >
bool File::read_impl( Type* buffer, std::streamsize elements )
{
#ifdef HAVE_CUDA
   const std::streamsize host_buffer_size = std::min( FileGPUvsCPUTransferBufferSize / (std::streamsize) sizeof(Type), elements );
   using BaseType = typename std::remove_cv< Type >::type;
   std::unique_ptr< BaseType[] > host_buffer{ new BaseType[ host_buffer_size ] };

   std::streamsize readElements = 0;
   while( readElements < elements )
   {
      const std::streamsize transfer = std::min( elements - readElements, host_buffer_size );
      file.read( reinterpret_cast<char*>(host_buffer.get()), sizeof(Type) * transfer );
      cudaMemcpy( (void*) &buffer[ readElements ],
                  (void*) host_buffer.get(),
                  transfer * sizeof( Type ),
                  cudaMemcpyHostToDevice );
      TNL_CHECK_CUDA_DEVICE;
      readElements += transfer;
   }
   return true;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

// MIC
template< typename Type,
          typename Device,
          typename, typename, typename >
bool File::read_impl( Type* buffer, std::streamsize elements )
{
#ifdef HAVE_MIC
   const std::streamsize host_buffer_size = std::min( FileGPUvsCPUTransferBufferSize / (std::streamsize) sizeof(Type), elements );
   using BaseType = typename std::remove_cv< Type >::type;
   std::unique_ptr< BaseType[] > host_buffer{ new BaseType[ host_buffer_size ] };

   std::streamsize readElements = 0;
   while( readElements < elements )
   {
      const std::streamsize transfer = std::min( elements - readElements, host_buffer_size );
      file.read( reinterpret_cast<char*>(host_buffer.get()), sizeof(Type) * transfer );

      Devices::MICHider<Type> device_buff;
      device_buff.pointer=buffer;
      #pragma offload target(mic) in(device_buff,readElements) in(host_buffer:length(transfer))
      {
         /*
         for(int i=0;i<transfer;i++)
              device_buff.pointer[readElements+i]=host_buffer[i];
          */
         memcpy(&(device_buff.pointer[readElements]), host_buffer.get(), transfer*sizeof(Type) );
      }

      readElements += transfer;
   }
   free( host_buffer );
   return true;
#else
   throw Exceptions::MICSupportMissing();
#endif
}

template< class Type, typename Device >
bool File::write( const Type* buffer, std::streamsize elements )
{
   TNL_ASSERT_GE( elements, 0, "Number of elements to write must be non-negative." );

   if( ! elements )
      return true;

   return write_impl< Type, Device >( buffer, elements );
}

// Host
template< typename Type,
          typename Device,
          typename >
bool File::write_impl( const Type* buffer, std::streamsize elements )
{
   file.write( reinterpret_cast<const char*>(buffer), sizeof(Type) * elements );
   return true;
}

// Cuda
template< typename Type,
          typename Device,
          typename, typename >
bool File::write_impl( const Type* buffer, std::streamsize elements )
{
#ifdef HAVE_CUDA
   const std::streamsize host_buffer_size = std::min( FileGPUvsCPUTransferBufferSize / (std::streamsize) sizeof(Type), elements );
   using BaseType = typename std::remove_cv< Type >::type;
   std::unique_ptr< BaseType[] > host_buffer{ new BaseType[ host_buffer_size ] };

   std::streamsize writtenElements = 0;
   while( writtenElements < elements )
   {
      const std::streamsize transfer = std::min( elements - writtenElements, host_buffer_size );
      cudaMemcpy( (void*) host_buffer.get(),
                  (void*) &buffer[ writtenElements ],
                  transfer * sizeof(Type),
                  cudaMemcpyDeviceToHost );
      TNL_CHECK_CUDA_DEVICE;
      file.write( reinterpret_cast<const char*>(host_buffer.get()), sizeof(Type) * transfer );
      writtenElements += transfer;
   }
   return true;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

// MIC
template< typename Type,
          typename Device,
          typename, typename, typename >
bool File::write_impl( const Type* buffer, std::streamsize elements )
{
#ifdef HAVE_MIC
   const std::streamsize host_buffer_size = std::min( FileGPUvsCPUTransferBufferSize / (std::streamsize) sizeof(Type), elements );
   using BaseType = typename std::remove_cv< Type >::type;
   std::unique_ptr< BaseType[] > host_buffer{ new BaseType[ host_buffer_size ] };

   std::streamsize writtenElements = 0;
   while( this->writtenElements < elements )
   {
      const std::streamsize transfer = std::min( elements - writtenElements, host_buffer_size );

      Devices::MICHider<const Type> device_buff;
      device_buff.pointer=buffer;
      #pragma offload target(mic) in(device_buff,writtenElements) out(host_buffer:length(transfer))
      {
         //THIS SHOULD WORK... BUT NOT WHY?
         /*for(int i=0;i<transfer;i++)
              host_buffer[i]=device_buff.pointer[writtenElements+i];
          */

         memcpy(host_buffer.get(), &(device_buff.pointer[writtenElements]), transfer*sizeof(Type) );
      }

      file.write( reinterpret_cast<const char*>(host_buffer.get()), sizeof(Type) * transfer );
      writtenElements += transfer;
   }
   return true;
#else
   throw Exceptions::MICSupportMissing();
#endif
}

inline bool fileExists( const String& fileName )
{
   std::fstream file;
   file.open( fileName.getString(), std::ios::in );
   return ! file.fail();
}


// serialization of strings
inline File& operator<<( File& file, const std::string& str )
{
   const int len = str.size();
   if( ! file.write( &len ) )
      throw Exceptions::FileSerializationError( getType< int >(), file.getFileName() );
   if( ! file.write( str.c_str(), len ) )
      throw Exceptions::FileSerializationError( "String", file.getFileName() );
   return file;
}

// deserialization of strings
inline File& operator>>( File& file, std::string& str )
{
   int length;
   if( ! file.read( &length ) )
      throw Exceptions::FileDeserializationError( getType< int >(), file.getFileName() );
   char buffer[ length ];
   if( length && ! file.read( buffer, length ) )
      throw Exceptions::FileDeserializationError( "String", file.getFileName() );
   str.assign( buffer, length );
   return file;
}

} // namespace TNL
