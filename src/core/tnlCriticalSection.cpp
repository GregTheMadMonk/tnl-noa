/***************************************************************************
                          tnlCriticalSection.cpp  -  description
                             -------------------
    begin                : Feb 20, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <errno.h>
#include <iostream>
#include <debug/tnlDebug.h>
#include <core/tnlCriticalSection.h>

using namespace std;

tnlCriticalSection :: tnlCriticalSection( key_t sectionKey,
                                          bool sectionMaster,
                                          int initialSemaphoreValue )
: semaphoreId( 0 ),
  sectionMaster( sectionMaster ),
  sectionKey( sectionKey ),
  sectionOk( true )
{
   dbgFunctionName( "tnlCriticalSection", "tnlCriticalSection");

   /****
    * Get the semaphore first.
    */
   dbgCout( "The critical section is getting a semaphore..." );
   semaphoreId = semget( sectionKey, 1, 0666 | IPC_CREAT );
   if( semaphoreId < 0 )
   {
      cerr << "I am not able to create new semaphore for IPC ( errno = " << errno << ")." << endl;
      sectionOk = false;
      return;
   }

   if( sectionMaster )
   {
      /****
       * Initialize the semaphore.
       */
      dbgCout( "The critical section is initializing the semaphore..." );
      union semun {
         int              val;
         struct semid_ds *buf;
         unsigned short  *array;
         struct seminfo  *__buf;
      } semaphoreUnion;

      semaphoreUnion. val = initialSemaphoreValue;
      if( semctl( semaphoreId, 0, SETVAL, semaphoreUnion ) == -1 )
      {
         cerr << "I am not able to set new semaphore for IPC ( errno = " << errno << ")." << endl;
         sectionOk = false;
      }
   }

}

tnlCriticalSection :: operator bool () const
{
   return sectionOk;
}

bool tnlCriticalSection :: enter()
{
   dbgFunctionName( "tnlCriticalSection", "enter");
   dbgCout( "The process is entering the critical section with semid " << semaphoreId << "..." );
   sembuf semBuf;
   semBuf. sem_num = 0;
   semBuf. sem_op = -1;
   semBuf. sem_flg = SEM_UNDO;
   if( semop( semaphoreId, &semBuf, 1 ) ==  -1 )
   {
      cerr << "The process cannot enter the critical section ( errno = " << errno << ")." << endl;
      return false;
   }
   return true;
}

bool tnlCriticalSection :: leave()
{
   dbgFunctionName( "tnlCriticalSection", "leave");
   dbgCout( "The process is leaving the critical section with semid " << semaphoreId << " ..." );
   sembuf semBuf;
   semBuf. sem_num = 0;
   semBuf. sem_op = 1;
   semBuf. sem_flg = SEM_UNDO;
   if( semop( semaphoreId, &semBuf, 1 ) ==  -1 )
   {
      cerr << "The process cannot leave the critical section ( errno = " << errno << ")." << endl;
      return false;
   }
   return true;
}

tnlCriticalSection :: ~tnlCriticalSection()
{
   dbgFunctionName( "tnlCriticalSection", "~tnlCriticalSection");
   if( sectionMaster )
   {
      /****
       * Release the semaphore.
       */
      dbgCout( "The critical section is releasing the semaphore ..." );
      union semun {
         int              val;
         struct semid_ds *buf;
         unsigned short  *array;
         struct seminfo  *__buf;
      } semaphoreUnion;
      if( semctl( semaphoreId, 0, IPC_RMID, semaphoreUnion ) == -1 )
      {
         cerr << "I am not able to release the semaphore ( errno = " << errno << ")." << endl;
      }
   }
}
