/***************************************************************************
                          tnlCriticalSection.h  -  description
                             -------------------
    begin                : Feb 20, 2011
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

#ifndef TNLCRITICALSECTION_H_
#define TNLCRITICALSECTION_H_

#include <sys/sem.h>

/*!***
 * This class wraps semaphor used for entering and leaving critical section in IPC.
 */
class tnlCriticalSection
{
   public:

   /*!**
    * This is a constructor taking a unique key for the semaphore.
    * @param sectionKey is a unique key for the enwrapped semaphore
    * @param sectionMaster one of the processes using this critical section must
    *        be a master and release the semaphore at the end.
    * @param initialSemaphoreValue is the initial value of the sempahore
    */
   tnlCriticalSection( key_t sectionKey,
                       bool sectionMaster,
                       int initalSemaphoreValue = 1 );

   operator bool () const;

   /*!**
    * Call this method to enter the section.
    */
   bool enter();

   /*!**
    * Call this method to leave the section.
    */
   bool leave();

   /*!**
    * Destructor for the critical section.
    */
   ~tnlCriticalSection();

   protected:

   key_t sectionKey;

   bool sectionMaster;

   int semaphoreId;

   bool sectionOk;
};
#endif /* TNLCRITICALSECTION_H_ */
