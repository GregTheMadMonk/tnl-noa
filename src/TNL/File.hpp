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
#include <ios>
#include <sstream>

#include <TNL/File.h>
#include <TNL/Assert.h>
#include <TNL/Exceptions/CudaSupportMissing.h>
#include <TNL/Exceptions/MICSupportMissing.h>
#include <TNL/Exceptions/FileSerializationError.h>
#include <TNL/Exceptions/FileDeserializationError.h>

namespace TNL {

inline File::Mode operator|( File::Mode m1, File::Mode m2 );

inline bool operator&( File::Mode m1, File::Mode m2 );

inline void File::open( const String& fileName, Mode mode )
{
   // enable exceptions
   file.exceptions( std::fstream::failbit | std::fstream::badbit | std::fstream::eofbit );

   close();

   auto ios_mode = std::ios::binary;
   if( mode & Mode::In ) ios_mode |= std::ios::in;
   if( mode & Mode::Out ) ios_mode |= std::ios::out;
   if( mode & Mode::Append ) ios_mode |= std::ios::app;
   if( mode & Mode::AtEnd ) ios_mode |= std::ios::ate;
   if( mode & Mode::Truncate ) ios_mode |= std::ios::trunc;
   try
   {
      file.open( fileName.getString(), ios_mode );
   }
   catch( std::ios_base::failure& )
   {
      std::stringstream msg;
      msg <<  "Unable to open file " << fileName << " ";
      if( mode & Mode::In )
         msg << " for reading.";
      if( mode & Mode::Out )
         msg << " for writing.";

      throw std::ios_base::failure( msg.str() );
   }

   this->fileName = fileName;
}

inline void File::close()
{
   if( file.is_open() )
   {
      try
      {
         file.close();
      }
      catch( std::ios_base::failure& )
      {
         std::stringstream msg;
         msg <<  "Unable to close file " << fileName << ".";

         throw std::ios_base::failure( msg.str() );
      }
   }
   // reset file name
   fileName = "";
}

template< typename Type,
          typename SourceType,
          typename Device >
void File::load( Type* buffer, std::streamsize elements )
{
   TNL_ASSERT_GE( elements, 0, "Number of elements to load must be non-negative." );

   if( ! elements )
      return;

   load_impl< Type, SourceType, Device >( buffer, elements );
}

// Host
template< typename Type,
          typename SourceType,
          typename Device,
          typename >
void File::load_impl( Type* buffer, std::streamsize elements )
{
   if( std::is_same< Type, SourceType >::value )
      file.read( reinterpret_cast<char*>(buffer), sizeof(Type) * elements );
   else
   {
      const std::streamsize cast_buffer_size = std::min( TransferBufferSize / (std::streamsize) sizeof(SourceType), elements );
      using BaseType = typename std::remove_cv< SourceType >::type;
      std::unique_ptr< BaseType[] > cast_buffer{ new BaseType[ cast_buffer_size ] };
      std::streamsize readElements = 0;
      while( readElements < elements )
      {
         const std::streamsize transfer = std::min( elements - readElements, cast_buffer_size );
         file.read( reinterpret_cast<char*>(cast_buffer.get()), sizeof(SourceType) * transfer );
         for( std::streamsize i = 0; i < transfer; i++ )
            buffer[ readElements ++ ] = static_cast< Type >( cast_buffer[ i ] );
         readElements += transfer;
      }
   }
}

// Cuda
template< typename Type,
          typename SourceType,
          typename Device,
          typename, typename >
void File::load_impl( Type* buffer, std::streamsize elements )
{
#ifdef HAVE_CUDA
   const std::streamsize host_buffer_size = std::min( TransferBufferSize / (std::streamsize) sizeof(Type), elements );
   using BaseType = typename std::remove_cv< Type >::type;
   std::unique_ptr< BaseType[] > host_buffer{ new BaseType[ host_buffer_size ] };

   std::streamsize readElements = 0;
   if( std::is_same< Type, SourceType >::value )
   {
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
   }
   else
   {
      const std::streamsize cast_buffer_size = std::min( TransferBufferSize / (std::streamsize) sizeof(SourceType), elements );
      using BaseType = typename std::remove_cv< SourceType >::type;
      std::unique_ptr< BaseType[] > cast_buffer{ new BaseType[ cast_buffer_size ] };

      while( readElements < elements )
      {
         const std::streamsize transfer = std::min( elements - readElements, cast_buffer_size );
         file.read( reinterpret_cast<char*>(cast_buffer.get()), sizeof(SourceType) * transfer );
         for( std::streamsize i = 0; i < transfer; i++ )
            host_buffer[ i ] = static_cast< Type >( cast_buffer[ i ] );
         cudaMemcpy( (void*) &buffer[ readElements ],
                     (void*) host_buffer.get(),
                     transfer * sizeof( Type ),
                     cudaMemcpyHostToDevice );
         TNL_CHECK_CUDA_DEVICE;
         readElements += transfer;
      }
   }
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

// MIC
template< typename Type,
          typename SourceType,
          typename Device,
          typename, typename, typename >
void File::load_impl( Type* buffer, std::streamsize elements )
{
#ifdef HAVE_MIC
   const std::streamsize host_buffer_size = std::min( TransferBufferSize / (std::streamsize) sizeof(Type), elements );
   using BaseType = typename std::remove_cv< Type >::type;
   std::unique_ptr< BaseType[] > host_buffer{ new BaseType[ host_buffer_size ] };

   std::streamsize readElements = 0;
   if( std::is_same< Type, SourceType >::value )
   {
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
   }
   else
   {
      std::cerr << "Type conversion during loading is not implemented for MIC." << std::endl;
      abort();
   }
#else
   throw Exceptions::MICSupportMissing();
#endif
}

template< typename Type,
          typename TargetType,
          typename Device >
void File::save( const Type* buffer, std::streamsize elements )
{
   TNL_ASSERT_GE( elements, 0, "Number of elements to save must be non-negative." );

   if( ! elements )
      return;

   save_impl< Type, TargetType, Device >( buffer, elements );
}

// Host
template< typename Type,
          typename TargetType,
          typename Device,
          typename >
void File::save_impl( const Type* buffer, std::streamsize elements )
{
   if( std::is_same< Type, TargetType >::value )
      file.write( reinterpret_cast<const char*>(buffer), sizeof(Type) * elements );
   else
   {
      const std::streamsize cast_buffer_size = std::min( TransferBufferSize / (std::streamsize) sizeof(TargetType), elements );
      using BaseType = typename std::remove_cv< TargetType >::type;
      std::unique_ptr< BaseType[] > cast_buffer{ new BaseType[ cast_buffer_size ] };
      std::streamsize writtenElements = 0;
      while( writtenElements < elements )
      {
         const std::streamsize transfer = std::min( elements - writtenElements, cast_buffer_size );
         for( std::streamsize i = 0; i < transfer; i++ )
            cast_buffer[ i ] = static_cast< TargetType >( buffer[ writtenElements ++ ] );
         file.write( reinterpret_cast<char*>(cast_buffer.get()), sizeof(TargetType) * transfer );
         writtenElements += transfer;
      }

   }
}

// Cuda
template< typename Type,
          typename TargetType,
          typename Device,
          typename, typename >
void File::save_impl( const Type* buffer, std::streamsize elements )
{
#ifdef HAVE_CUDA
   const std::streamsize host_buffer_size = std::min( TransferBufferSize / (std::streamsize) sizeof(Type), elements );
   using BaseType = typename std::remove_cv< Type >::type;
   std::unique_ptr< BaseType[] > host_buffer{ new BaseType[ host_buffer_size ] };

   std::streamsize writtenElements = 0;
   if( std::is_same< Type, TargetType >::value )
   {
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
   }
   else
   {
      const std::streamsize cast_buffer_size = std::min( TransferBufferSize / (std::streamsize) sizeof(TargetType), elements );
      using BaseType = typename std::remove_cv< TargetType >::type;
      std::unique_ptr< BaseType[] > cast_buffer{ new BaseType[ cast_buffer_size ] };

      while( writtenElements < elements )
      {
         const std::streamsize transfer = std::min( elements - writtenElements, host_buffer_size );
         cudaMemcpy( (void*) host_buffer.get(),
                     (void*) &buffer[ writtenElements ],
                     transfer * sizeof(Type),
                     cudaMemcpyDeviceToHost );
         TNL_CHECK_CUDA_DEVICE;
         for( std::streamsize i = 0; i < transfer; i++ )
            cast_buffer[ i ] = static_cast< TargetType >( host_buffer[ i ] );

         file.write( reinterpret_cast<const char*>(cast_buffer.get()), sizeof(TargetType) * transfer );
         writtenElements += transfer;
      }
   }
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

// MIC
template< typename Type,
          typename TargetType,
          typename Device,
          typename, typename, typename >
void File::save_impl( const Type* buffer, std::streamsize elements )
{
#ifdef HAVE_MIC
   const std::streamsize host_buffer_size = std::min( TransferBufferSize / (std::streamsize) sizeof(Type), elements );
   using BaseType = typename std::remove_cv< Type >::type;
   std::unique_ptr< BaseType[] > host_buffer{ new BaseType[ host_buffer_size ] };

   std::streamsize writtenElements = 0;
   if( std::is_same< Type, TargetType >::value )
   {
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
   }
   else
   {
      std::cerr << "Type conversion during saving is not implemented for MIC." << std::endl;
      abort();
   }
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
   try
   {
       file.save( &len );
   }
   catch(...)
   {
      throw Exceptions::FileSerializationError( getType< int >(), file.getFileName() );
   }
   try
   {
      file.save( str.c_str(), len );
   }
   catch(...)
   {
      throw Exceptions::FileSerializationError( "String", file.getFileName() );
   }
   return file;
}

// deserialization of strings
inline File& operator>>( File& file, std::string& str )
{
   int length;
   try
   {
      file.load( &length );
   }
   catch(...)
   {
      throw Exceptions::FileDeserializationError( getType< int >(), file.getFileName() );
   }
   char buffer[ length ];
   if( length )
   {
      try
      {
         file.load( buffer, length );
      }
      catch(...)
      {
         throw Exceptions::FileDeserializationError( "String", file.getFileName() );
      }
   }
   str.assign( buffer, length );
   return file;
}

inline File::Mode operator|( File::Mode m1, File::Mode m2 )
{
   return static_cast< File::Mode >( static_cast< int >( m1 ) | static_cast< int >( m2 ) );
}

inline bool operator&( File::Mode m1, File::Mode m2 )
{
   return static_cast< bool >( static_cast< int >( m1 ) & static_cast< int >( m2 ) );
}

} // namespace TNL
