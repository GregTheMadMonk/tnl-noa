/***************************************************************************
                          tnlSharedMemory.h  -  description
                             -------------------
    begin                : Feb 21, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLSHAREDMEMORY_H_
#define TNLSHAREDMEMORY_H_

#include <errno.h>
#include <sys/shm.h>

/*!***
 * This template wraps shared memory for IPC. For the communication
 * in MPI style we usually allocate the shared memory which is accessed by
 * several processes. It is good to synchronize them by counting number
 * accesses to the memory. For this reason we bind two counters with the
 * allocated memory. For example for the broadcast operation we know that
 * we must write the broadcasted data once and read them once for each process
 * except the broadcasting one. Therefore the receiving data will wait for the
 * broadcasting one to increase the writing counter. Then each process copy the
 * data to its memory and increase the reading counter. When each process have read
 * its data we can free the shared memory. This template does not care about any
 * critical sections. This must be managed elsewhere.
 */
template< typename DataType, typename Index = int >
class tnlSharedMemory
{
   public:

   tnlSharedMemory( key_t sharedMemoryKey, Index size, bool master );

   operator bool () const;

   const DataType* getData() const;

   DataType* getData();

   int getReadingCounter() const;

   int getWritingCounter() const;

   void setReadingCounter( int value );

   void setWritingCounter( int value );

   void increaseReadingCounter();

   void increaseWritingCounter();

   ~tnlSharedMemory();

   protected:

   int sharedMemoryId;

   void* sharedMemory;

   int* readingCounter;

   int* writingCounter;

   DataType* data;

   bool isOk;

   //! If this is true this process will erase allocated memory segment.
   bool sharedMemoryMaster;
};

template< typename DataType, typename Index >
tnlSharedMemory< DataType, Index > :: tnlSharedMemory( key_t sharedMemoryKey, Index size, bool master )
: sharedMemoryId( 0 ),
  sharedMemory( 0 ),
  readingCounter( 0 ),
  writingCounter( 0 ),
  isOk( false ),
  sharedMemoryMaster( master )
{
   dbgFunctionName( "tnlSharedMemory","tnlSharedMemory" );
   /****
    * We first allocate the shared memory.
    * The key must be unique for each group of processes accessing this memory.
    * We allocate one more byte which we use
    * as a flag telling if the data are written and ready for reading.
    */

   dbgCout( "Allocating the shared memory with a key " << sharedMemoryKey );
   sharedMemoryId = shmget( sharedMemoryKey,
                            sizeof( DataType ) * size + 1 + 2 * sizeof( int),
                            0666 | IPC_CREAT );
   if( sharedMemoryId == -1 )
   {
      cerr << "Sorry, I was not able to allocate the shared memory for the IPC ( errno = " << errno << ")." << endl;
      isOk = false;
      return;
   }

   /****
    * Now we attach to the shared memory
    */
   sharedMemory = shmat( sharedMemoryId,
                         ( void* ) 0,
                         0 );
   if( sharedMemory == ( void* ) -1 )
   {
      cerr << "Sorry, I was not able to attach to the shared memory for the IPC ( errno = " << errno << ")." << endl;
      isOk = false;
      return;
   }
   int* sharedMemoryInt = ( int* ) sharedMemory;
   readingCounter = sharedMemoryInt ++;
   writingCounter = sharedMemoryInt ++;
   data = ( DataType* ) sharedMemoryInt;
   isOk = true;
}

template< typename DataType, typename Index >
tnlSharedMemory< DataType, Index > :: operator bool () const
{
   return isOk;
}

template< typename DataType, typename Index >
const DataType* tnlSharedMemory< DataType, Index > :: getData() const
{
   tnlAssert( data != 0, );
   return data;
}

template< typename DataType, typename Index >
DataType* tnlSharedMemory< DataType, Index > :: getData()
{
   tnlAssert( data != 0, );
   return data;
}

template< typename DataType, typename Index >
int tnlSharedMemory< DataType, Index > :: getReadingCounter() const
{
   tnlAssert( readingCounter != 0, );
   return * readingCounter;
}

template< typename DataType, typename Index >
int tnlSharedMemory< DataType, Index > :: getWritingCounter() const
{
   tnlAssert( writingCounter != 0, );
   return * writingCounter;
}

template< typename DataType, typename Index >
void tnlSharedMemory< DataType, Index > :: setReadingCounter( int value )
{
   tnlAssert( readingCounter != 0, );
   *readingCounter = value;
}

template< typename DataType, typename Index >
void tnlSharedMemory< DataType, Index > :: setWritingCounter( int value )
{
   tnlAssert( writingCounter != 0, );
   *writingCounter = value;
}

template< typename DataType, typename Index >
void tnlSharedMemory< DataType, Index > :: increaseReadingCounter()
{
   tnlAssert( readingCounter != 0, );
   ( * readingCounter ) ++;
}

template< typename DataType, typename Index >
void tnlSharedMemory< DataType, Index > :: increaseWritingCounter()
{
   tnlAssert( writingCounter != 0, );
   ( *writingCounter ) ++;
}

template< typename DataType, typename Index >
tnlSharedMemory< DataType, Index > :: ~tnlSharedMemory()
{
   dbgFunctionName( "tnlSharedMemory","~tnlSharedMemory" );
   /****
    * Release the shared memory.
    */
   dbgCout( "Releasing the shared memory..." );
   if( shmdt( sharedMemory ) == -1 )
   {
      cerr << "Sorry I am not able to detach the shared memory after sending data ( errno = " << errno << ")." << endl;
   }
   if( sharedMemoryMaster )
   {
      dbgCout( "Deleting the shared memory..." );
      if( shmctl( sharedMemoryId, IPC_RMID, 0 ) == -1 )
         cerr << "I am not able to delete allocated shared memory ( errno = " << errno << ")." << endl;
   }
}


#endif /* TNLSHAREDMEMORY_H_ */
