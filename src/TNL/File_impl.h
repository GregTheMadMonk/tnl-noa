/***************************************************************************
                          tnlFile_impl.h  -  description
                             -------------------
    begin                : Mar 5, Oct 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>

#include <TNL/File.h>
#include <TNL/Exceptions/CudaSupportMissing.h>
#include <TNL/Exceptions/MICSupportMissing.h>

namespace TNL {


template< typename Type, typename Device >
bool File::read( Type* buffer )
{
   return read< Type, Device, int >( buffer, 1 );
}

template< typename Type, typename Device >
bool File::write( const Type* buffer )
{
   return write< Type, Device, int >( buffer, 1 );
}

template< typename Type, typename Device, typename Index >
bool File::read( Type* buffer,
                 const Index& _elements )
{
   TNL_ASSERT_GE( _elements, 0, "Number of elements to read must be non-negative." );

   // convert _elements from Index to std::size_t, which is *unsigned* type
   // (expected by fread etc)
   std::size_t elements = (std::size_t) _elements;

   if( ! elements )
      return true;
   if( ! fileOK )
   {
      std::cerr << "File " << fileName << " was not properly opened. " << std::endl;
      return false;
   }
   if( mode != IOMode::read )
   {
      std::cerr << "File " << fileName << " was not opened for reading. " << std::endl;
      return false;
   }

   return read_impl< Type, Device >( buffer, elements );
}

// Host
template< typename Type,
          typename Device,
          typename >
bool File::read_impl( Type* buffer,
                      const std::size_t& elements )
{
   this->readElements = 0;
   if( std::fread( buffer,
                   sizeof( Type ),
                   elements,
                   file ) != elements )
   {
      std::cerr << "I am not able to read the data from the file " << fileName << "." << std::endl;
      std::perror( "Fread ended with the error code" );
      return false;
   }
   this->readElements = elements;
   return true;
}

// Cuda
template< typename Type,
          typename Device,
          typename, typename >
bool File::read_impl( Type* buffer,
                      const std::size_t& elements )
{
#ifdef HAVE_CUDA
   this->readElements = 0;
   const std::size_t host_buffer_size = std::min( tnlFileGPUvsCPUTransferBufferSize / sizeof( Type ), elements );
   using BaseType = typename std::remove_cv< Type >::type;
   BaseType* host_buffer = new BaseType[ host_buffer_size ];

   while( readElements < elements )
   {
      std::size_t transfer = std::min( elements - readElements, host_buffer_size );
      std::size_t transfered = std::fread( host_buffer, sizeof( Type ), transfer, file );
      if( transfered != transfer )
      {
         std::cerr << "I am not able to read the data from the file " << fileName << "." << std::endl;
         std::cerr << transfered << " bytes were transfered. " << std::endl;
         std::perror( "Fread ended with the error code" );
         delete[] host_buffer;
         return false;
      }

      cudaMemcpy( ( void* ) & ( buffer[ readElements ] ),
                  host_buffer,
                  transfer * sizeof( Type ),
                  cudaMemcpyHostToDevice );
      if( ! TNL_CHECK_CUDA_DEVICE )
      {
         std::cerr << "Transfer of data from the CUDA device to the file " << this->fileName
              << " failed." << std::endl;
         delete[] host_buffer;
         return false;
      }
      this->readElements += transfer;
   }
   delete[] host_buffer;
   return true;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

// MIC
template< typename Type,
          typename Device,
          typename, typename, typename >
bool File::read_impl( Type* buffer,
                      const std::size_t& elements )
{
#ifdef HAVE_MIC
   this->readElements = 0;
   const std::size_t host_buffer_size = std::min( tnlFileGPUvsCPUTransferBufferSize / sizeof( Type ), elements );
   Type * host_buffer = (Type *)malloc( sizeof( Type ) * host_buffer_size );
   readElements = 0;
   if( ! host_buffer )
   {
      std::cerr << "I am sorry but I cannot allocate supporting buffer on the host for writing data from the GPU to the file "
                << this->getFileName() << "." << std::endl;
      return false;
   }

   while( readElements < elements )
   {
      int transfer = std::min(  elements - readElements , host_buffer_size );
      size_t transfered = fread( host_buffer, sizeof( Type ), transfer, file );
      if( transfered != transfer )
      {
         std::cerr << "I am not able to read the data from the file " << fileName << "." << std::endl;
         std::cerr << transfered << " bytes were transfered. " << std::endl;
         perror( "Fread ended with the error code" );
         return false;
      }
      Devices::MICHider<Type> device_buff;
      device_buff.pointer=buffer;
      #pragma offload target(mic) in(device_buff,readElements) in(host_buffer:length(transfer))
      {
         /*
         for(int i=0;i<transfer;i++)
              device_buff.pointer[readElements+i]=host_buffer[i];
          */
         memcpy(&(device_buff.pointer[readElements]),host_buffer, transfer*sizeof(Type) );
      }

      readElements += transfer;
   }
   free( host_buffer );
   return true;
#else
   throw Exceptions::MICSupportMissing();
#endif
}

template< class Type, typename Device, typename Index >
bool File::write( const Type* buffer,
                  const Index _elements )
{
   TNL_ASSERT_GE( _elements, 0, "Number of elements to write must be non-negative." );

   // convert _elements from Index to std::size_t, which is *unsigned* type
   // (expected by fread etc)
   std::size_t elements = (std::size_t) _elements;

   if( ! elements )
      return true;
   if( ! fileOK )
   {
      std::cerr << "File " << fileName << " was not properly opened. " << std::endl;
      return false;
   }
   if( mode != IOMode::write )
   {
      std::cerr << "File " << fileName << " was not opened for writing. " << std::endl;
      return false;
   }

   return write_impl< Type, Device >( buffer, elements );
}

// Host
template< typename Type,
          typename Device,
          typename >
bool File::write_impl( const Type* buffer,
                       const std::size_t& elements )
{
   this->writtenElements = 0;
   if( std::fwrite( buffer,
                    sizeof( Type ),
                    elements,
                    this->file ) != elements )
   {
      std::cerr << "I am not able to write the data to the file " << fileName << "." << std::endl;
      std::perror( "Fwrite ended with the error code" );
      return false;
   }
   this->writtenElements = elements;
   return true;
}

// Cuda
template< typename Type,
          typename Device,
          typename, typename >
bool File::write_impl( const Type* buffer,
                       const std::size_t& elements )
{
#ifdef HAVE_CUDA
   this->writtenElements = 0;
   const std::size_t host_buffer_size = std::min( tnlFileGPUvsCPUTransferBufferSize / sizeof( Type ),
                                             elements );
   using BaseType = typename std::remove_cv< Type >::type;
   BaseType* host_buffer = new BaseType[ host_buffer_size ];

   while( this->writtenElements < elements )
   {
      std::size_t transfer = std::min( elements - this->writtenElements, host_buffer_size );
      cudaMemcpy( host_buffer,
                  ( void* ) & ( buffer[ this->writtenElements ] ),
                  transfer * sizeof( Type ),
                  cudaMemcpyDeviceToHost );
      if( ! TNL_CHECK_CUDA_DEVICE )
      {
         std::cerr << "Transfer of data from the file " << this->fileName
              << " to the CUDA device failed." << std::endl;
         delete[] host_buffer;
         return false;
      }
      if( std::fwrite( host_buffer,
                       sizeof( Type ),
                       transfer,
                       this->file ) != transfer )
      {
         std::cerr << "I am not able to write the data to the file " << fileName << "." << std::endl;
         std::perror( "Fwrite ended with the error code" );
         delete[] host_buffer;
         return false;
      }
      this->writtenElements += transfer;
   }
   delete[] host_buffer;
   return true;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

// MIC
template< typename Type,
          typename Device,
          typename, typename, typename >
bool File::write_impl( const Type* buffer,
                       const std::size_t& elements )
{
#ifdef HAVE_MIC
   this->writtenElements = 0;
   const std::size_t host_buffer_size = std::min( tnlFileGPUvsCPUTransferBufferSize / sizeof( Type ),
                                                  elements );
   Type * host_buffer = (Type *)malloc( sizeof( Type ) * host_buffer_size );
   if( ! host_buffer )
   {
      std::cerr << "I am sorry but I cannot allocate supporting buffer on the host for writing data from the GPU to the file "
                << this->getFileName() << "." << std::endl;
      return false;
   }

   while( this->writtenElements < elements )
   {
       std::size_t transfer = std::min( elements - this->writtenElements, host_buffer_size );

      Devices::MICHider<const Type> device_buff;
      device_buff.pointer=buffer;
      #pragma offload target(mic) in(device_buff,writtenElements) out(host_buffer:length(transfer))
      {
         //THIS SHOULD WORK... BUT NOT WHY?
         /*for(int i=0;i<transfer;i++)
              host_buffer[i]=device_buff.pointer[writtenElements+i];
          */

         memcpy(host_buffer,&(device_buff.pointer[writtenElements]), transfer*sizeof(Type) );
      }

      if( fwrite( host_buffer,
                  sizeof( Type ),
                  transfer,
                  this->file ) != transfer )
      {
         std::cerr << "I am not able to write the data to the file " << fileName << "." << std::endl;
         perror( "Fwrite ended with the error code" );
         return false;
      }
      this->writtenElements += transfer;
   }
   free( host_buffer );
   return true;
#else
   throw Exceptions::MICSupportMissing();
#endif
}

} // namespace TNL
