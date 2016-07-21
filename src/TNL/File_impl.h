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
                      const Index& elements )
{
   Assert( elements >= 0,
              std::cerr << " elements = " << elements << std::endl; );
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
   const Index host_buffer_size = std::min( ( Index ) ( tnlFileGPUvsCPUTransferBufferSize / sizeof( Type ) ),
                                          elements );
   void* host_buffer( 0 );
   if( Device :: getDeviceType() == "tnlHost" )
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
         std::cerr << "I am sorry but I cannot allocate supporting buffer on the host for writing data from the GPU to the file "
              << this->getFileName() << "." << std::endl;
         return false;

      }

      while( readElements < elements )
      {
         int transfer = :: min( ( Index ) ( elements - readElements ), host_buffer_size );
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
      tnlCudaSupportMissingMessage;;
      return false;
#endif
   }
   return true;
};

template< class Type, typename Device, typename Index >
bool File ::  write( const Type* buffer,
                        const Index elements )
{
   Assert( elements >= 0,
              std::cerr << " elements = " << elements << std::endl; );
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
   const long int host_buffer_size = std::min( ( Index ) ( tnlFileGPUvsCPUTransferBufferSize / sizeof( Type ) ),
                                          elements );
   if( Device :: getDeviceType() == "tnlHost" )
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
            std::cerr << "I am sorry but I cannot allocate supporting buffer on the host for writing data from the GPU to the file "
                 << this->getFileName() << "." << std::endl;
            return false;
         }

         while( this->writtenElements < elements )
         {
            Index transfer = min( elements - this->writtenElements, host_buffer_size );
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
         tnlCudaSupportMissingMessage;;
         return false;
#endif
   }
   return true;
};

} // namespace TNL


