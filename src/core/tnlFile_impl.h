/***************************************************************************
                          tnlFile_impl.h  -  description
                             -------------------
    begin                : Mar 5, Oct 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLFILE_IMPL_H
#define	TNLFILE_IMPL_H

template< typename Type, typename Device >
bool tnlFile :: read( Type* buffer )
{
   return read< Type, Device, int >( buffer, 1 );
};

template< typename Type, typename Device >
bool tnlFile :: write( const Type* buffer )
{
   return write< Type, Device, int >( buffer, 1 );
};


template< typename Type, typename Device, typename Index >
bool tnlFile :: read( Type* buffer,
                      const Index& elements )
{
   tnlAssert( elements >= 0,
              cerr << " elements = " << elements << endl; );
   if( ! elements )
      return true;
   if( ! fileOK )
   {
      cerr << "File " << fileName << " was not properly opened. " << endl;
      return false;
   }
   if( mode != tnlReadMode )
   {
      cerr << "File " << fileName << " was not opened for reading. " << endl;
      return false;
   }
   this->readElements = 0;
   const Index host_buffer_size = :: Min( ( Index ) ( tnlFileGPUvsCPUTransferBufferSize / sizeof( Type ) ),
                                          elements );
   void* host_buffer( 0 );
   if( Device :: getDeviceType() == "tnlHost" )
   {
      if( fread( buffer,
             sizeof( Type ),
             elements,
             file ) != elements )
      {
         cerr << "I am not able to read the data from the file " << fileName << "." << endl;
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
         cerr << "I am sorry but I cannot allocate supporting buffer on the host for writing data from the GPU to the file "
              << this->getFileName() << "." << endl;
         return false;

      }

      while( readElements < elements )
      {
         int transfer = :: Min( ( Index ) ( elements - readElements ), host_buffer_size );
         size_t transfered = fread( host_buffer, sizeof( Type ), transfer, file );
         if( transfered != transfer )
         {
            cerr << "I am not able to read the data from the file " << fileName << "." << endl;
            cerr << transfered << " bytes were transfered. " << endl;
            perror( "Fread ended with the error code" );
            return false;
         }

         cudaMemcpy( ( void* ) & ( buffer[ readElements ] ),
                     host_buffer,
                     transfer * sizeof( Type ),
                     cudaMemcpyHostToDevice );
         if( ! checkCudaDevice )
         {
            cerr << "Transfer of data from the CUDA device to the file " << this->fileName
                 << " failed." << endl;
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
bool tnlFile ::  write( const Type* buffer,
                        const Index elements )
{
   tnlAssert( elements >= 0,
              cerr << " elements = " << elements << endl; );
   if( ! elements )
      return true;
   if( ! fileOK )
   {
      cerr << "File " << fileName << " was not properly opened. " << endl;
      return false;
   }
   if( mode != tnlWriteMode )
   {
      cerr << "File " << fileName << " was not opened for writing. " << endl;
      return false;
   }

   Type* buf = const_cast< Type* >( buffer );
   void* host_buffer( 0 );
   this->writtenElements = 0;
   const long int host_buffer_size = :: Min( ( Index ) ( tnlFileGPUvsCPUTransferBufferSize / sizeof( Type ) ),
                                          elements );
   if( Device :: getDeviceType() == "tnlHost" )
   {
      if( fwrite( buf,
                  sizeof( Type ),
                  elements,
                  this->file ) != elements )
      {
         cerr << "I am not able to write the data to the file " << fileName << "." << endl;
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
            cerr << "I am sorry but I cannot allocate supporting buffer on the host for writing data from the GPU to the file "
                 << this->getFileName() << "." << endl;
            return false;
         }

         while( this->writtenElements < elements )
         {
            Index transfer = Min( elements - this->writtenElements, host_buffer_size );
            cudaMemcpy( host_buffer,
                       ( void* ) & ( buffer[ this->writtenElements ] ),
                       transfer * sizeof( Type ),
                       cudaMemcpyDeviceToHost );
            if( ! checkCudaDevice )
            {
               cerr << "Transfer of data from the file " << this->fileName
                    << " to the CUDA device failed." << endl;
               free( host_buffer );
               return false;
            }
            if( fwrite( host_buffer,
                        sizeof( Type ),
                        transfer,
                        this->file ) != transfer )
            {
               cerr << "I am not able to write the data to the file " << fileName << "." << endl;
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



#endif	/* TNLFILE_IMPL_H */

