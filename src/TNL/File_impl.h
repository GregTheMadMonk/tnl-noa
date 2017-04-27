/***************************************************************************
                          tnlFile_impl.h  -  description
                             -------------------
    begin                : Mar 5, Oct 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

namespace TNL {


template< typename Type, typename Device >
bool File :: read( Type* buffer )
{
   return read< Type, Device, int >( buffer, 1 );
};

template< typename Type, typename Device >
bool File :: write( const Type* buffer )
{
   return write< Type, Device, int >( buffer, 1 );
};

template< typename Type, typename Device, typename Index >
bool File :: read( Type* buffer,
                   const Index& _elements )
{
   TNL_ASSERT( _elements >= 0,
           std::cerr << " elements = " << _elements << std::endl; );

   // convert _elements from Index to size_t, which is *unsigned* type
   // (expected by fread etc)
   size_t elements = (size_t) _elements;

   if( ! elements )
      return true;
   if( ! fileOK )
   {
      std::cerr << "File " << fileName << " was not properly opened. " << std::endl;
      return false;
   }
   if( mode != tnlReadMode )
   {
      std::cerr << "File " << fileName << " was not opened for reading. " << std::endl;
      return false;
   }
   this->readElements = 0;
   const size_t host_buffer_size = std::min( tnlFileGPUvsCPUTransferBufferSize / sizeof( Type ),
                                             elements );
   void* host_buffer( 0 );
   if( std::is_same< Device, Devices::Host >::value )
   {
      if( fread( buffer,
             sizeof( Type ),
             elements,
             file ) != elements )
      {
         std::cerr << "I am not able to read the data from the file " << fileName << "." << std::endl;
         perror( "Fread ended with the error code" );
         return false;
      }
      this->readElements = elements;
      return true;
   }
   if( std::is_same< Device, Devices::Cuda >::value )
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
         std::cerr << "I am sorry but I cannot allocate supporting buffer on the host for writing data from the GPU to the file "
              << this->getFileName() << "." << std::endl;
         return false;

      }

      while( readElements < elements )
      {
         size_t transfer = std::min( elements - readElements, host_buffer_size );
         size_t transfered = fread( host_buffer, sizeof( Type ), transfer, file );
         if( transfered != transfer )
         {
            std::cerr << "I am not able to read the data from the file " << fileName << "." << std::endl;
            std::cerr << transfered << " bytes were transfered. " << std::endl;
            perror( "Fread ended with the error code" );
            return false;
         }

         cudaMemcpy( ( void* ) & ( buffer[ readElements ] ),
                     host_buffer,
                     transfer * sizeof( Type ),
                     cudaMemcpyHostToDevice );
         if( ! checkCudaDevice )
         {
            std::cerr << "Transfer of data from the CUDA device to the file " << this->fileName
                 << " failed." << std::endl;
            free( host_buffer );
            return false;
         }
         readElements += transfer;
      }
      free( host_buffer );
      return true;
#else
      CudaSupportMissingMessage;;
      return false;
#endif
   }
   
   //MIC
   if( std::is_same< Device, Devices::MIC >::value )
   {
#ifdef HAVE_MIC           
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
#endif
   }
   
   return true;
};

template< class Type, typename Device, typename Index >
bool File :: write( const Type* buffer,
                    const Index _elements )
{
   TNL_ASSERT( _elements >= 0,
           std::cerr << " elements = " << _elements << std::endl; );

   // convert _elements from Index to size_t, which is *unsigned* type
   // (expected by fread etc)
   size_t elements = (size_t) _elements;

   if( ! elements )
      return true;
   if( ! fileOK )
   {
      std::cerr << "File " << fileName << " was not properly opened. " << std::endl;
      return false;
   }
   if( mode != tnlWriteMode )
   {
      std::cerr << "File " << fileName << " was not opened for writing. " << std::endl;
      return false;
   }

   Type* buf = const_cast< Type* >( buffer );
   void* host_buffer( 0 );
   this->writtenElements = 0;
   const size_t host_buffer_size = std::min( tnlFileGPUvsCPUTransferBufferSize / sizeof( Type ),
                                             elements );
   if( std::is_same< Device, Devices::Host >::value )
   {
      if( fwrite( buf,
                  sizeof( Type ),
                  elements,
                  this->file ) != elements )
      {
         std::cerr << "I am not able to write the data to the file " << fileName << "." << std::endl;
         perror( "Fwrite ended with the error code" );
         return false;
      }
      this->writtenElements = elements;
      return true;
   }
   if( std::is_same< Device, Devices::Cuda >::value )
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
            std::cerr << "I am sorry but I cannot allocate supporting buffer on the host for writing data from the GPU to the file "
                 << this->getFileName() << "." << std::endl;
            return false;
         }

         while( this->writtenElements < elements )
         {
            size_t transfer = std::min( elements - this->writtenElements, host_buffer_size );
            cudaMemcpy( host_buffer,
                       ( void* ) & ( buffer[ this->writtenElements ] ),
                       transfer * sizeof( Type ),
                       cudaMemcpyDeviceToHost );
            if( ! checkCudaDevice )
            {
               std::cerr << "Transfer of data from the file " << this->fileName
                    << " to the CUDA device failed." << std::endl;
               free( host_buffer );
               return false;
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
         CudaSupportMissingMessage;;
         return false;
#endif
   }
   //MIC
   if( std::is_same< Device, Devices::MIC >::value )
   {
#ifdef HAVE_MIC
         Type * host_buffer = (Type *)malloc( sizeof( Type ) * host_buffer_size );
         if( ! host_buffer )
         {
            std::cerr << "I am sorry but I cannot allocate supporting buffer on the host for writing data from the GPU to the file "
                 << this->getFileName() << "." << std::endl;
            return false;
         }

         while( this->writtenElements < elements )
         {
            Index transfer = std::min( elements - this->writtenElements, host_buffer_size );
            
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
#endif
   } 
   return true;
};

} // namespace TNL


