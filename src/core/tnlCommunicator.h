/***************************************************************************
                          tnlCommunicator.h  -  description
                             -------------------
    begin                : Feb 5, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLCOMMUNICATOR_H_
#define TNLCOMMUNICATOR_H_

#include <sys/types.h>
#include <sys/wait.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <errno.h>
#include <unistd.h>
#include <debug/tnlDebug.h>
#include <core/vectors/tnlVector.h>
#include <core/tnlCriticalSection.h>
#include <core/tnlSharedMemory.h>

const key_t tnlIPCKey = 76025938;

enum tnlReductionOperation
{
   tnlSumReduction,
   tnlProductReduction,
   tnlMinReduction,
   tnlMaxReduction
};

//! This object establishes basic communication operations in TNL.
/*!***
 * It is aimed for the communication between GPUs and inside the MPI cluster.
 * To make the communication in TNL simple we adopt the MPI philosophy to
 * make use it for the communication between GPUs. In case of GPUs, this communicator
 * creates new process for each GPU device (except of the first one). Then we need
 * to communicate between processes. It is very similar to MPI. Therefore it makes good
 * sense to wrap the MPI communication to this communicator too. This may simplify
 * combining communication between GPUs and cluster nodes in the GPU clusters.
 * What devices will communicate by this communicator depends on the template
 * parameter @param Device.
 */
template< int Dimensions, typename Device >
class tnlCommunicator
{
   public:

   tnlCommunicator();

   //! This method initialize the group size.
   /*!***
    * The parameter @param maxCommunicationGroupSize says the maximum number
    * of processes running in this communicator. If it is zero, the group size
    * will be set to the greatest possible number. For example in case of the GPU
    * communication, the group size will equal to the number of the GPUs installed
    * in the system. In the case of MPI communication, the group size will equal
    * to the number of the processes started by mpirun.
    */
   bool setCommunicationGroupSize( int maxCommunicationGroupSize = 0 );

   int getCommunicationGroupSize() const;

   bool setDimensions( const tnlTuple< Dimensions, int >& dimensions );

   const tnlTuple< Dimensions, int >& getDimensions() const;

   const tnlTuple< Dimensions, int >& getNodeCoordinates() const;

   int getDeviceId() const;

   //! This method starts the communicator.
   /*!***
    * In case of the GPU communication it creates new processes so that there
    * is one process for each GPU in the system. Then the device IDs are set.
    * In the case of the MPI, only the device IDs are set.
    */
   bool start();

   bool stop();

   template< typename DataType >
   bool send( const DataType* data,
              int destinationId ) const;

   template< typename DataType >
   bool receive( DataType* data,
                 int sourceId ) const;

   template< typename DataType, typename Index >
   bool send( const tnlVector< DataType, Device, Index >& data,
              int destinationId ) const;

   template< typename DataType, typename Index >
   bool receive( tnlVector< DataType, Device, Index >& data,
                 int sourceId ) const;

   template< typename DataType >
   bool broadcast( DataType* data, int sourceId );

   template< typename DataType, typename Index >
   bool broadcast( tnlVector< DataType, Device, Index >& data, int sourceId );

   template< typename DataType >
   bool reduction( DataType* data, tnlReductionOperation operation, int targetId );

   template< typename DataType, typename Index >
   bool scatter( const tnlVector< DataType, Device, Index >& inputData,
                 tnlVector< DataType, Device, Index >& scatteredData,
                 int sourceId );

   template< typename DataType, typename Index >
   bool gather( const tnlVector< DataType, Device, Index >& inputData,
                tnlVector< DataType, Device, Index >& gatheredData,
                int targetId );

   bool barrier();


   ~tnlCommunicator();

   protected:

   int communicationGroupSize;

   tnlTuple< Dimensions, int > dimensions;

   tnlTuple< Dimensions, int > nodeCoordinates;

   int deviceID;

   pid_t processID;
};

template< int Dimensions, typename Device >
tnlCommunicator< Dimensions, Device > :: tnlCommunicator()
: communicationGroupSize( 0 ),
  deviceID( -1 ),
  processID( -1 )
{
}

template< int Dimensions, typename Device >
int tnlCommunicator< Dimensions, Device > :: getCommunicationGroupSize() const
{
   return this -> communicationGroupSize;
}

template< int Dimensions, typename Device >
bool tnlCommunicator< Dimensions, Device > :: setDimensions( const tnlTuple< Dimensions, int >& dimensions )
{
   this -> dimensions = dimensions;
   // TODO: add automatic dimensions setting from the group size
}

template< int Dimensions, typename Device >
const tnlTuple< Dimensions, int >& tnlCommunicator< Dimensions, Device > :: getDimensions() const
{
   return this -> dimensions;
}

template< int Dimensions, typename Device >
const tnlTuple< Dimensions, int >& tnlCommunicator< Dimensions, Device > :: getNodeCoordinates() const
{
   return this -> nodeCoordinates;
}

template< int Dimensions, typename Device >
int tnlCommunicator< Dimensions, Device > :: getDeviceId() const
{
    return this -> deviceID;
}

template< int Dimensions, typename Device >
bool tnlCommunicator< Dimensions, Device > :: setCommunicationGroupSize( int communicationGroupSize )
{
    this -> communicationGroupSize = communicationGroupSize;
    // TODO: add automatic groupsize setting
}

template< int Dimensions, typename Device >
bool tnlCommunicator< Dimensions, Device > :: start()
{
   dbgFunctionName( "tnlCommunicator", "start" );
   if( this -> getCommunicationGroupSize() < 1 )
      this -> setCommunicationGroupSize();
   if( this -> getCommunicationGroupSize() < 1 )
   {
      cerr << "Sorry, but I have wrong size ( " << this -> getCommunicationGroupSize() << " of the communication group. I cannot create a communicator." << endl;
      return false;
   }
   if( this -> getDimensions() == tnlTuple< Dimensions, int >( 0 ) )
      this -> setDimensions( tnlTuple< Dimensions, int >( 0 ) );
   if( this -> getDimensions() == tnlTuple< Dimensions, int >( -1 ) )
   {
      cerr << "Sorry, but I have wrong dimensions ( " << this -> getDimensions() << " of the communication group. I cannot create a communicator." << endl;
      return false;
   }
   if( Device :: getDeviceType() == "tnlCuda" ||
       Device :: getDeviceType() == "tnlHost" )
   {
      deviceID = 0;
      int currentDeviceID = 1;
      for( int i = 0; i < this -> getCommunicationGroupSize() - 1; i ++ )
      {
         pid_t child_pid = fork();
         if( child_pid != 0 )
         {
            /****
             * This is a parent process.
             */
            currentDeviceID ++;
            dbgCout( "The MASTER process is setting ID to " << currentDeviceID );
         }
         else
         {
            deviceID = currentDeviceID;
            dbgCout( "The CHILD process is setting ID to " << deviceID );
            return true;
         } // if( child_pid != 0 ) - else
      } // for( int i = 0; i < this -> getCommunicationGroupSize() - 1; i ++ )
   }
}

template< int Dimensions, typename Device >
bool tnlCommunicator< Dimensions, Device > :: stop()
{
   dbgFunctionName( "tnlCommunicator", "stop")
   if( deviceID == 0 )
   {
      for( int i = 1; i < getCommunicationGroupSize(); i ++ )
      {
         int childStatus;
         wait( &childStatus );
      }
   }
   else
   {
      dbgCout( "Exiting process for the device " << deviceID );
      exit( EXIT_SUCCESS );
   }

}

template< int Dimensions, typename Device >
template< typename DataType >
bool tnlCommunicator< Dimensions, Device > :: send( const DataType* data,
                                        int destinationID ) const
{
   dbgFunctionName( "tnlCommunicator", "send" );
   /****
    * We first allocate the shared memory.
    */
   dbgExpr( getDeviceId() );
   key_t currentIPCKey = tnlIPCKey + deviceID + 1;
   dbgCout( "The sending process is allocating the shared memory with a key " << currentIPCKey );
   tnlSharedMemory< DataType > sharedMemory( currentIPCKey, 1, true );
   if( ! sharedMemory )
      return false;

   /****
    * We must reset the shared memory counters.
    * This must be done in a critical section.
    * We initiate the critical section with value 0, which means that it is closed
    * until this process opens it.
    */
   dbgCout( "The sending process is creating a critical section..." );
   tnlCriticalSection criticalSection( currentIPCKey, true, 0 );
   if( ! criticalSection )
   {
      cerr << "The sending process with the ID " << getDeviceId() << " cannot create a critical section." << endl;
      return false;
   }

   dbgCout( "The sending process is the shared memory counters to zero ..." )
   sharedMemory. setReadingCounter( 0 );
   sharedMemory. setWritingCounter( 0 );

   dbgCout( "The sending process is leaving the critical section ..." )
   if( ! criticalSection. leave() )
   {
      cerr << "The process " << getDeviceId() << " cannot leave the critical section to send the data." << endl;
      return false;
   }
   /****
    * Now we copy the data to the shared memory.
    * The DataType cannot be a type with the dynamic memory allocation.
    */
   if( Device :: getDevice() == tnlHostDevice )
   {
      memcpy( ( void* ) sharedMemory. getData(),
              data,
              sizeof( DataType ) );
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {

   }
   dbgCout( "The sending process has copied data " << * ( ( DataType* ) sharedMemory. getData() ) );

   dbgCout( "The sending process is increasing the writing counter of the shared memory ..." )
   sharedMemory. increaseWritingCounter();

   dbgCout( "The sending process is waiting for the receiving process to read the data ..." )
   while( sharedMemory. getReadingCounter() != 1 )
      usleep( 1 );
   return true;
}

template< int Dimensions, typename Device >
template< typename DataType >
bool tnlCommunicator< Dimensions, Device > :: receive( DataType* data,
                                           int sourceID ) const
{
   dbgFunctionName( "tnlCommunicator", "receive" );
   /****
    * We first allocate the shared memory.
    */
   key_t currentIPCKey = tnlIPCKey + sourceID + 1;
   dbgCout( "The receiving process is allocating the shared memory with a key " << currentIPCKey );
   tnlSharedMemory< DataType > sharedMemory( currentIPCKey, 1, false );
   if( ! sharedMemory )
      return false;

   /****
    * We will now wait for the sender to reset the shared memory counters.
    * This is done in a critical section.
    */
   dbgCout( "The receiving process is creating a critical section..." );
   tnlCriticalSection criticalSection( currentIPCKey, false );
   if( ! criticalSection )
   {
      cerr << "The receiving process cannot create a critical section." << endl;
      return false;
   }

   dbgCout( "The receiving process is entering the critical section..." );
   if( ! criticalSection. enter() )
   {
      cerr << "The process " << deviceID << " cannot enter the critical section to receive data." << endl;
      return false;
   }

   dbgCout( "The receiving process is leaving the critical section..." );
   if( ! criticalSection. leave() )
   {
      cerr << "The process " << deviceID << " cannot leave the critical section to receive data." << endl;
      return false;
   }

   /****
    * Now wait for the data to be copied to the shared memory.
    */
   dbgCout( "The receiving process is wating for the data to be copied..." );
   while( sharedMemory. getWritingCounter() != 1 )
      usleep( 1 );
   /****
    * Now we copy the data from the shared memory.
    * The DataType cannot by a type with the dynamic memory allocation.
    */
   dbgCout( "The receiving process is copying the data from the shared memory " << *( sharedMemory. getData() ) );
   memcpy( ( void* ) data,
           ( void* ) sharedMemory. getData(),
           sizeof( DataType ) );
   /****
    * We may now simply write 2 to the flag byte...
    */
   sharedMemory. increaseReadingCounter();
   return true;
}


template< int Dimensions, typename Device >
template< typename DataType, typename Index >
bool tnlCommunicator< Dimensions, Device > :: send( const tnlVector< DataType, Device, Index >& data,
                                        int destinationID ) const
{
   dbgFunctionName( "tnlCommunicator", "send" );
   /****
    * We first allocate the shared memory.
    */
   dbgExpr( getDeviceId() );
   key_t currentIPCKey = tnlIPCKey + getDeviceId() + 1;
   dbgCout( "The sending process is allocating the shared memory with a key " << currentIPCKey );
   tnlSharedMemory< DataType > sharedMemory( currentIPCKey, data. getSize(), true );
   if( ! sharedMemory )
      return false;

   /****
    * We will now reset the shred memory accesses counters.
    * This must be done in a critical section.
    * We initiate it as closed so no other process can enter it before
    * this and before this one opens it.
    */
   dbgCout( "The sending process is creating a critical section..." );
   tnlCriticalSection criticalSection( currentIPCKey, true, 0 );
   if( ! criticalSection )
   {
      cerr << "The sending process with the ID " << getDeviceId() << " cannot create a critical section." << endl;
      return false;
   }
   dbgCout( "The sending process is setting the flag byte to tnlSharedMemoryReady ..." )
   sharedMemory. setReadingCounter( 0 );
   sharedMemory. setWritingCounter( 0 );

   dbgCout( "The sending process is leaving the critical section ..." )
   if( ! criticalSection. leave() )
   {
      cerr << "The process " << getDeviceId() << " cannot leave the critical section to send the data." << endl;
      return false;
   }
   /****
    * Now we copy the data to the shared memory.
    * The DataType cannot be a type with the dynamic memory allocation.
    */
   tnlSharedArray< DataType, tnlHost, Index > sendingBuffer( "sendingBuffer" );
   sendingBuffer. bind( sharedMemory. getData(), data. getSize() );
   sendingBuffer = data;
   dbgCout( "The sending process is setting the flag byte to tnlSharedMemoryDataSent ..." )
   sharedMemory. increaseWritingCounter();

   dbgCout( "The sending process is waiting for the receiving process to read the data ..." )
   while( sharedMemory. getReadingCounter() != 1 )
      usleep( 1 );

   return true;
}

template< int Dimensions, typename Device >
template< typename DataType, typename Index >
bool tnlCommunicator< Dimensions, Device > :: receive( tnlVector< DataType, Device, Index >& data,
                                           int sourceID ) const
{
   dbgFunctionName( "tnlCommunicator", "receive" );
   /****
    * We first allocate the shared memory.
    */
   key_t currentIPCKey = tnlIPCKey + sourceID + 1;
   dbgCout( "The receiving process is allocating the shared memory with a key " << currentIPCKey );
   tnlSharedMemory< DataType > sharedMemory( currentIPCKey, data. getSize(), false );
   if( ! sharedMemory )
      return false;

   /****
    * We will now wait for the sender to reset the shared memory counters.
    * This is done in a critical section.
    */
   dbgCout( "The receiving process is creating a critical section..." );
   tnlCriticalSection criticalSection( currentIPCKey, false );
   if( ! criticalSection )
   {
      cerr << "The receiving process cannot create a critical section." << endl;
      return false;
   }

   dbgCout( "The receiving process is entering the critical section..." );
   if( ! criticalSection. enter() )
   {
      cerr << "The process " << deviceID << " cannot enter the critical section to receive data." << endl;
      return false;
   }

   dbgCout( "The receiving process is leaving the critical section..." );
   if( ! criticalSection. leave() )
   {
      cerr << "The process " << deviceID << " cannot leave the critical section to receive data." << endl;
      return false;
   }

   dbgCout( "The receiving process is waiting for the data to be copied..." );
   while( sharedMemory. getWritingCounter() != 1 )
      usleep( 1 );

   dbgCout( "The receiving process is copying the data from the shared memory..." );
   tnlSharedArray< DataType, tnlHost, Index > receivingBuffer( "receivingBuffer" );
   receivingBuffer. bind( sharedMemory. getData(), data. getSize() );
   data = receivingBuffer;

   dbgCout( "The receiving process is increasing the reading counter of the shared memory..." );
   sharedMemory. increaseReadingCounter();
   return true;
}

template< int Dimensions, typename Device >
template< typename DataType >
bool tnlCommunicator< Dimensions, Device > :: broadcast( DataType* data, int sourceId )
{
   dbgFunctionName( "tnlCommunicator", "broadcast" );
   /****
    * We first allocate the shared memory.
    */
   key_t currentIPCKey = tnlIPCKey + sourceId + 1;
   dbgCout( "The processes are allocating the shared memory with a key " << currentIPCKey );
   bool sharedMemoryMaster( false );
   if( getDeviceId() == sourceId )
      sharedMemoryMaster = true;
   tnlSharedMemory< DataType > sharedMemory( currentIPCKey, 1, sharedMemoryMaster );

   /****
    * Reset the shared memory counters
    */
   dbgCout( "The process " << getDeviceId() << " is creating a critical section..." );
   tnlCriticalSection criticalSection( currentIPCKey,
                                       getDeviceId() == sourceId,
                                       0 );
   if( ! criticalSection )
   {
      cerr << "The process with the ID " << getDeviceId() << " cannot create a critical section." << endl;
      return false;
   }


   if( getDeviceId() != sourceId )
   {
      dbgCout( "The process " << getDeviceId() << " is entering the critical section..." )
      if( ! criticalSection. enter() )
      {
         cerr << "The process " << getDeviceId() << " cannot enter the critical section to send the data." << endl;
         return false;
      }
   }
   else
   {
      dbgCout( "The broadcasting process " << getDeviceId() << " is resetting the shared memory counters ..." );
      sharedMemory. setReadingCounter( 0 );
      sharedMemory. setWritingCounter( 0 );
   }

   dbgCout( "The process " << getDeviceId() << " is leaving the critical section ..." )
   if( ! criticalSection. leave() )
   {
      cerr << "The process " << getDeviceId() << " cannot leave the critical section to send the data." << endl;
      return false;
   }

   /****
    * The broadcasting process will now write data to the shared memory.
    * After that it increases the writing counter of the shared memory.
    * The other processes waits until the writing counter equals onr.
    */
   if( getDeviceId() == sourceId )
   {
      /****
       * Now we copy the data to the shared memory.
       * The DataType cannot be a type with the dynamic memory allocation.
       */
      if( Device :: getDevice() == tnlHostDevice )
      {
         memcpy( ( void* ) ( sharedMemory. getData() ),
                 ( void* ) data,
                 sizeof( DataType ) );
      }
      if( Device :: getDevice() == tnlCudaDevice )
      {

      }
      dbgCout( "The broadcasting process " << getDeviceId() << " has copied data " << * ( ( DataType* ) sharedMemory. getData() ) );

      dbgCout( "The broadcasting process " << getDeviceId() << " is entering the critical section..." );
      if( ! criticalSection. enter() )
      {
         cerr << "The process " << getDeviceId() << " cannot enter the critical section to receive data." << endl;
         return false;
      }

      dbgCout( "The broadcasting process " << getDeviceId() << " is increasing the writing counter of the shared memory ..." )
      sharedMemory. increaseWritingCounter();

      dbgCout( "The broadcasting process " << getDeviceId() << " is leaving the critical section..." );
      if( ! criticalSection. leave() )
      {
         cerr << "The process  " << getDeviceId() << " cannot leave the critical section to receive data." << endl;
         return false;
      }

      dbgCout( "The broadcasting process  " << getDeviceId() << " is waiting for the receiving process to read the data ..." )
      while( sharedMemory. getReadingCounter() != communicationGroupSize - 1 )
         usleep( 1 );
      return true;
   }
   else
   {
      /****
       * The process waits for the broadcasting process to reset the shared memory counters.
       */
      dbgCout( "The receiving process  " << getDeviceId() << " is waiting for the data to be copied..." );
      while( sharedMemory. getWritingCounter() != 1 )
         usleep( 1 );

      dbgCout( "The receiving process  " << getDeviceId() << " is copying the data from the shared memory " << *( DataType* ) sharedMemory. getData() );
      memcpy( ( void* ) data,
              ( void* ) ( sharedMemory. getData() ),
              sizeof( DataType ) );

      /****
       * Increase the reading counter (inside the critical section)
       */
      dbgCout( "The receiving process  " << getDeviceId() << " is entering the critical section..." );
      if( ! criticalSection. enter() )
      {
         cerr << "The process  " << getDeviceId() << " cannot enter the critical section to receive data." << endl;
         return false;
      }

      dbgCout( "The receiving process " << getDeviceId() << " is increasing the reading counter to " << sharedMemory. getReadingCounter() + 1 << "..." );
      sharedMemory. increaseReadingCounter();

      dbgCout( "The receiving process " << getDeviceId() << " is leaving the critical section..." );
      if( ! criticalSection. leave() )
      {
         cerr << "The process  " << getDeviceId() << " cannot leave the critical section to receive data." << endl;
         return false;
      }
      return true;
   }
}

template< int Dimensions, typename Device >
template< typename DataType, typename Index >
bool tnlCommunicator< Dimensions, Device > :: broadcast( tnlVector< DataType, Device, Index >& data,
                                             int sourceId )
{
   dbgFunctionName( "tnlCommunicator", "broadcast" );
   /****
    * We first allocate the shared memory.
    */
   key_t currentIPCKey = tnlIPCKey + sourceId + 1;
   dbgCout( "The process " << getDeviceId() << " is allocating the shared memory with a key " << currentIPCKey );
   bool sharedMemoryMaster( false );
   if( getDeviceId() == sourceId )
      sharedMemoryMaster = true;
   tnlSharedMemory< DataType > sharedMemory( currentIPCKey, 1, sharedMemoryMaster );

   /****
    * Reset the shared memory counters
    */
   dbgCout( "The process" << getDeviceId() << " is creating a critical section..." );
   tnlCriticalSection criticalSection( currentIPCKey,
                                       getDeviceId() == sourceId,
                                       0 );
   if( ! criticalSection )
   {
      cerr << "The process with the ID " << getDeviceId() << " cannot create a critical section." << endl;
      return false;
   }

   if( getDeviceId() != sourceId )
   {
      dbgCout( "The process " << getDeviceId() << " is entering the critical section..." )
      if( ! criticalSection. enter() )
      {
         cerr << "The process " << getDeviceId() << " cannot enter the critical section to send the data." << endl;
         return false;
      }
   }
   else
   {
      dbgCout( "The broadcasting process is resetting the shared memory counters ..." );
      sharedMemory. setReadingCounter( 0 );
      sharedMemory. setWritingCounter( 0 );
   }

   dbgCout( "The process " << getDeviceId() << " is leaving the critical section ..." )
   if( ! criticalSection. leave() )
   {
      cerr << "The process " << getDeviceId() << " cannot leave the critical section to send the data." << endl;
      return false;
   }

   /****
    * The broadcasting process will now write data to the shared memory.
    * After that it increases the writing counter of the shared memory.
    * The other processes waits until the writing counter equals onr.
    */
   if( getDeviceId() == sourceId )
   {
      /****
       * Now we copy the data to the shared memory.
       * The DataType cannot be a type with the dynamic memory allocation.
       */
      tnlSharedArray< DataType, tnlHost, Index > sendingBuffer( "sendingBuffer" );
      sendingBuffer. bind( sharedMemory. getData(), data. getSize() );
      sendingBuffer = data;

      dbgCout( "The broadcasting process " << getDeviceId() << " has copied data." );
      dbgCout( "The broadcasting process " << getDeviceId() << " is increasing the writing counter of the shared memory ..." )

      dbgCout( "The broadcasting process " << getDeviceId() << " is entering the critical section..." );
      if( ! criticalSection. enter() )
      {
         cerr << "The process " << deviceID << " cannot enter the critical section to receive data." << endl;
         return false;
      }
      sharedMemory. increaseWritingCounter();
      dbgCout( "The broadcasting process " << getDeviceId() << " is leaving the critical section..." );
      if( ! criticalSection. leave() )
      {
         cerr << "The process " << deviceID << " cannot leave the critical section to receive data." << endl;
         return false;
      }

      dbgCout( "The broadcasting process " << getDeviceId() << " is waiting for the receiving process to read the data ..." )
      while( sharedMemory. getReadingCounter() != communicationGroupSize - 1 )
         usleep( 1 );
   }
   else
   {
      dbgCout( "The receiving process " << getDeviceId() << " is waiting for the data to be copied..." );
      while( sharedMemory. getWritingCounter() != 1 )
         usleep( 1 );

      dbgCout( "The receiving process " << getDeviceId() << " copying the data from the shared memory. " );
      tnlSharedArray< DataType, tnlHost, Index > receivingBuffer( "receivingBuffer" );
      receivingBuffer. bind( sharedMemory. getData(), data. getSize() );
      data = receivingBuffer;

      /****
       * Increase the reading counter (inside the critical section)
       */
      dbgCout( "The receiving process " << getDeviceId() << " is entering the critical section..." );
      if( ! criticalSection. enter() )
      {
         cerr << "The process " << deviceID << " cannot enter the critical section to receive data." << endl;
         return false;
      }

      dbgCout( "The receiving process " << getDeviceId() << " is increasing the reading counter..." );
      sharedMemory. increaseReadingCounter();

      dbgCout( "The receiving process " << getDeviceId() << " is leaving the critical section..." );
      if( ! criticalSection. leave() )
      {
         cerr << "The process " << deviceID << " cannot leave the critical section to receive data." << endl;
         return false;
      }
   }
   dbgCout( "The process " << getDeviceId() << " is finishing the broadcasting operation.")
   return true;
}

template< int Dimensions, typename Device >
template< typename DataType >
bool tnlCommunicator< Dimensions, Device > :: reduction( DataType* data, tnlReductionOperation operation, int targetId )
{
   dbgFunctionName( "tnlCommunicator", "reduction" );
   /****
    * We first allocate the shared memory.
    */
   key_t currentIPCKey = tnlIPCKey + targetId + 1;
   dbgCout( "The processes are allocating the shared memory with a key " << currentIPCKey );
   tnlSharedMemory< DataType > sharedMemory( currentIPCKey,
                                             getCommunicationGroupSize(),
                                             getDeviceId() == targetId );

   /****
    * Reset the shared memory counters
    */
   dbgCout( "The process " << getDeviceId() << " is creating a critical section..." );
   tnlCriticalSection criticalSection( currentIPCKey,
                                       getDeviceId() == targetId,
                                       0 );
   if( ! criticalSection )
   {
      cerr << "The process " << getDeviceId() << " cannot create a critical section." << endl;
      return false;
   }


   if( getDeviceId() != targetId )
   {
      dbgCout( "The process " << getDeviceId() << " is entering the critical section..." )
      if( ! criticalSection. enter() )
      {
         cerr << "The process " << getDeviceId() << " cannot enter the critical section to send the data." << endl;
         return false;
      }
   }
   else
   {
      dbgCout( "The target process " << getDeviceId() << " is resetting the shared memory counters ..." );
      sharedMemory. setReadingCounter( 0 );
      sharedMemory. setWritingCounter( 0 );
   }

   dbgCout( "The process " << getDeviceId() << " is leaving the critical section ..." )
   if( ! criticalSection. leave() )
   {
      cerr << "The process " << getDeviceId() << " cannot leave the critical section to send the data." << endl;
      return false;
   }

   /****
    * All processes will now write their data to its position in the shared memory array.
    * After that each process increases the writing counter of the shared memory.
    * The target process waits until all processes has written their data which
    * means until the writing counter equals communication group size.
    */


   if( Device :: getDevice() == tnlHostDevice )
   {
      dbgCout( "The process " << getDeviceId() << " is writing data " << * data );
      memcpy( ( void* ) &( sharedMemory. getData()[ getDeviceId() ] ),
              ( void* ) data,
              sizeof( DataType ) );
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {

   }
   dbgCout( "The process " << getDeviceId() << " has copied data " << sharedMemory. getData()[ getDeviceId() ] );

   dbgCout( "The process " << getDeviceId() << " is entering the critical section..." );
   if( ! criticalSection. enter() )
   {
      cerr << "The process " << getDeviceId() << " cannot enter the critical section to receive data." << endl;
      return false;
   }

   dbgCout( "The process " << getDeviceId() << " is increasing the writing counter of the shared memory ..." )
   sharedMemory. increaseWritingCounter();

   dbgCout( "The process " << getDeviceId() << " is leaving the critical section..." );
   if( ! criticalSection. leave() )
   {
      cerr << "The process  " << getDeviceId() << " cannot leave the critical section to receive data." << endl;
      return false;
   }

   if( getDeviceId() == targetId )
   {
      dbgCout( "The target process " << getDeviceId() << " is waiting until the other processes write their data ..." )
      while( sharedMemory. getWritingCounter() != communicationGroupSize )
         usleep( 1 );
      dbgCout( "The target process " << getDeviceId() << " is performing the reduction ..." );
      DataType* array = sharedMemory. getData();
      DataType aux = array[ 0 ];
      for( int i = 1; i < getCommunicationGroupSize(); i ++ )
      {
         switch( operation )
         {
            case tnlSumReduction:
               aux += array[ i ];
               break;
            case tnlProductReduction:
               aux *= array[ i ];
               break;
            case tnlMinReduction:
               aux = Min( aux, array[ i ] );
               break;
            case tnlMaxReduction:
               aux = Max( aux, array[ i ] );
               break;
         }
      }
      if( Device :: getDevice() == tnlHostDevice )
      {
         *data = aux;
         dbgExpr( *data );
      }
      if( Device :: getDevice() == tnlCudaDevice )
      {

      }
   }
   return true;
}

template< int Dimensions, typename Device >
template< typename DataType, typename Index >
bool tnlCommunicator< Dimensions, Device > :: scatter( const tnlVector< DataType, Device, Index >& inputData,
                                           tnlVector< DataType, Device, Index >& scatteredData,
                                           int sourceId )
{
   dbgFunctionName( "tnlCommunicator", "scatter" );
   /****
    * We first allocate the shared memory.
    */
   key_t currentIPCKey = tnlIPCKey + sourceId + 1;
   dbgCout( "The process " << getDeviceId() << " is allocating the shared memory with a key " << currentIPCKey );
   tnlSharedMemory< DataType > sharedMemory( currentIPCKey,
                                             inputData. getSize(),
                                             getDeviceId() == sourceId );
   if( ! sharedMemory )
      return false;

   /****
    * Reset the shared memory counters
    */
   dbgCout( "The process" << getDeviceId() << " is creating a critical section..." );
   tnlCriticalSection criticalSection( currentIPCKey,
                                       getDeviceId() == sourceId,
                                       0 );
   if( ! criticalSection )
   {
      cerr << "The process with the ID " << getDeviceId() << " cannot create a critical section." << endl;
      return false;
   }

   if( getDeviceId() != sourceId )
   {
      dbgCout( "The process " << getDeviceId() << " is entering the critical section..." )
      if( ! criticalSection. enter() )
      {
         cerr << "The process " << getDeviceId() << " cannot enter the critical section to send the data." << endl;
         return false;
      }
   }
   else
   {
      dbgCout( "The master process is resetting the shared memory counters ..." );
      sharedMemory. setReadingCounter( 0 );
      sharedMemory. setWritingCounter( 0 );
   }

   dbgCout( "The process " << getDeviceId() << " is leaving the critical section ..." )
   if( ! criticalSection. leave() )
   {
      cerr << "The process " << getDeviceId() << " cannot leave the critical section to send the data." << endl;
      return false;
   }

   /****
    * The master process will now write data to the shared memory.
    * After that it increases the writing counter of the shared memory.
    * The other processes waits until the writing counter equals one.
    */
   if( getDeviceId() == sourceId )
   {
      /****
       * Now we copy the data to the shared memory.
       * The DataType cannot be a type with the dynamic memory allocation.
       */
      tnlSharedArray< DataType, tnlHost, Index > sendingBuffer( "sendingBuffer" );
      sendingBuffer. bind( sharedMemory. getData(), inputData. getSize() );
      sendingBuffer = inputData;

      dbgCout( "The broadcasting process " << getDeviceId() << " has copied data." );
      dbgCout( "The broadcasting process " << getDeviceId() << " is increasing the writing counter of the shared memory ..." )

      dbgCout( "The broadcasting process " << getDeviceId() << " is entering the critical section..." );
      if( ! criticalSection. enter() )
      {
         cerr << "The process " << deviceID << " cannot enter the critical section to receive data." << endl;
         return false;
      }
      sharedMemory. increaseWritingCounter();
      dbgCout( "The broadcasting process " << getDeviceId() << " is leaving the critical section..." );
      if( ! criticalSection. leave() )
      {
         cerr << "The process " << deviceID << " cannot leave the critical section to receive data." << endl;
         return false;
      }

      dbgCout( "The process " << getDeviceId() << " is copying the data from the shared memory. " );
      tnlSharedArray< DataType, tnlHost, Index > receivingBuffer( "receivingBuffer" );
      receivingBuffer. bind( sharedMemory. getData(), scatteredData. getSize() );
      scatteredData = receivingBuffer;

      dbgCout( "The broadcasting process " << getDeviceId() << " is waiting for the receiving process to read the data ..." )
      while( sharedMemory. getReadingCounter() != communicationGroupSize - 1 )
         usleep( 1 );
   }
   else
   {
      dbgCout( "The process " << getDeviceId() << " is waiting for the data to be copied..." );
      while( sharedMemory. getWritingCounter() != 1 )
         usleep( 1 );

      dbgCout( "The process " << getDeviceId() << " is copying the data from the shared memory. " );
      tnlSharedArray< DataType, tnlHost, Index > receivingBuffer( "receivingBuffer" );
      receivingBuffer. bind( sharedMemory. getData(), scatteredData. getSize() );
      scatteredData = receivingBuffer;

      /****
       * Increase the reading counter (inside the critical section)
       */
      dbgCout( "The process " << getDeviceId() << " is entering the critical section..." );
      if( ! criticalSection. enter() )
      {
         cerr << "The process " << deviceID << " cannot enter the critical section to receive data." << endl;
         return false;
      }

      dbgCout( "The process " << getDeviceId() << " is increasing the reading counter..." );
      sharedMemory. increaseReadingCounter();

      dbgCout( "The process " << getDeviceId() << " is leaving the critical section..." );
      if( ! criticalSection. leave() )
      {
         cerr << "The process " << deviceID << " cannot leave the critical section to receive data." << endl;
         return false;
      }
   }
   dbgCout( "The process " << getDeviceId() << " is finishing the scattering operation.")
   return true;
}

template< int Dimensions, typename Device >
template< typename DataType, typename Index >
bool tnlCommunicator< Dimensions, Device > :: gather( const tnlVector< DataType, Device, Index >& inputData,
                                           tnlVector< DataType, Device, Index >& gatheredData,
                                           int targetId )
{
   dbgFunctionName( "tnlCommunicator", "gather" );
   /****
    * We first allocate the shared memory.
    */
   key_t currentIPCKey = tnlIPCKey + targetId + 1;
   dbgCout( "The processes are allocating the shared memory with a key " << currentIPCKey );
   tnlSharedMemory< DataType > sharedMemory( currentIPCKey,
                                             getCommunicationGroupSize() * inputData. getSize(),
                                             getDeviceId() == targetId );

   /****
    * Reset the shared memory counters
    */
   dbgCout( "The process " << getDeviceId() << " is creating a critical section..." );
   tnlCriticalSection criticalSection( currentIPCKey,
                                       getDeviceId() == targetId,
                                       0 );
   if( ! criticalSection )
   {
      cerr << "The process " << getDeviceId() << " cannot create a critical section." << endl;
      return false;
   }


   if( getDeviceId() != targetId )
   {
      dbgCout( "The process " << getDeviceId() << " is entering the critical section..." )
      if( ! criticalSection. enter() )
      {
         cerr << "The process " << getDeviceId() << " cannot enter the critical section to send the data." << endl;
         return false;
      }
   }
   else
   {
      dbgCout( "The target process " << getDeviceId() << " is resetting the shared memory counters ..." );
      sharedMemory. setReadingCounter( 0 );
      sharedMemory. setWritingCounter( 0 );
   }

   dbgCout( "The process " << getDeviceId() << " is leaving the critical section ..." )
   if( ! criticalSection. leave() )
   {
      cerr << "The process " << getDeviceId() << " cannot leave the critical section to send the data." << endl;
      return false;
   }

   /****
    * All processes will now write their data to its position in the shared memory array.
    * After that each process increases the writing counter of the shared memory.
    * The target process waits until all processes has written their data which
    * means until the writing counter equals communication group size.
    */

   if( Device :: getDevice() == tnlHostDevice )
   {
      dbgCout( "The process " << getDeviceId() << " is writing data ... " );
      memcpy( ( void* ) &( sharedMemory. getData()[ getDeviceId() * inputData. getSize() ] ),
              ( void* ) inputData. getData(),
              sizeof( DataType ) );
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {

   }
   dbgCout( "The process " << getDeviceId() << " has copied data " << sharedMemory. getData()[ getDeviceId() ] );

   dbgCout( "The process " << getDeviceId() << " is entering the critical section..." );
   if( ! criticalSection. enter() )
   {
      cerr << "The process " << getDeviceId() << " cannot enter the critical section to receive data." << endl;
      return false;
   }

   dbgCout( "The process " << getDeviceId() << " is increasing the writing counter of the shared memory ..." )
   sharedMemory. increaseWritingCounter();

   dbgCout( "The process " << getDeviceId() << " is leaving the critical section..." );
   if( ! criticalSection. leave() )
   {
      cerr << "The process  " << getDeviceId() << " cannot leave the critical section to receive data." << endl;
      return false;
   }

   if( getDeviceId() == targetId )
   {
      dbgCout( "The target process " << getDeviceId() << " is waiting until the other processes write their data ..." )
      while( sharedMemory. getWritingCounter() != communicationGroupSize )
         usleep( 1 );
      dbgCout( "The target process " << getDeviceId() << " is gathering the data ..." );
      tnlSharedArray< DataType, tnlHost, Index > sharedMemoryVector( "sharedMemoryVector" );
      sharedMemoryVector. bind( sharedMemory. getData(), getCommunicationGroupSize() * inputData. getSize() );
      gatheredData = sharedMemoryVector;
   }
   return true;
}

template< int Dimensions, typename Device >
bool tnlCommunicator< Dimensions, Device > :: barrier()
{
   dbgFunctionName( "tnlCommunicator", "gather" );
   /****
    * We first allocate the shared memory.
    */
   key_t currentIPCKey = tnlIPCKey + 1;
   dbgCout( "The processes are allocating the shared memory with a key " << currentIPCKey );
   tnlSharedMemory< char > sharedMemory( currentIPCKey,
                                         1,
                                         getDeviceId() == 0 );

   /****
    * Reset the shared memory counters
    */
   dbgCout( "The process " << getDeviceId() << " is creating a critical section..." );
   tnlCriticalSection criticalSection( currentIPCKey,
                                       getDeviceId() == 0,
                                       0 );
   if( ! criticalSection )
   {
      cerr << "The process " << getDeviceId() << " cannot create a critical section." << endl;
      return false;
   }


   if( getDeviceId() != 0 )
   {
      dbgCout( "The process " << getDeviceId() << " is entering the critical section..." )
      if( ! criticalSection. enter() )
      {
         cerr << "The process " << getDeviceId() << " cannot enter the critical section to send the data." << endl;
         return false;
      }
   }
   else
   {
      dbgCout( "The target process " << getDeviceId() << " is resetting the shared memory counters ..." );
      sharedMemory. setReadingCounter( 0 );
   }

   dbgCout( "The process " << getDeviceId() << " is leaving the critical section ..." )
   if( ! criticalSection. leave() )
   {
      cerr << "The process " << getDeviceId() << " cannot leave the critical section to send the data." << endl;
      return false;
   }

   /****
    * Now each process will increase the read counter and then waits until
    * this counter reaches the number of all processes.
    */

   dbgCout( "The process " << getDeviceId() << " is entering the critical section..." )
   if( ! criticalSection. enter() )
   {
      cerr << "The process " << getDeviceId() << " cannot enter the critical section to send the data." << endl;
      return false;
   }
   sharedMemory. increaseReadingCounter();
   dbgCout( "The process " << getDeviceId() << " is leaving the critical section ..." )
   if( ! criticalSection. leave() )
   {
      cerr << "The process " << getDeviceId() << " cannot leave the critical section to send the data." << endl;
      return false;
   }

   while( sharedMemory. getReadingCounter() != getCommunicationGroupSize() )
      usleep( 1 );
}

template< int Dimensions, typename Device >
tnlCommunicator< Dimensions, Device > :: ~tnlCommunicator()
{
   stop();
}

#endif /* TNLCOMMUNICATOR_H_ */
