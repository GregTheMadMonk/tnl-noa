/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

/***************************************************************************
                          SmartPointersRegister.h  -  description
                             -------------------
    begin                : Apr 29, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#pragma once

#include <vector>
#include <list>
#include <TNL/SmartPointer.h>
#include <TNL/Assert.h>

class SmartPointersRegister
{   
  
   public:
   
      SmartPointersRegister( int devicesCount = 1 );
      
      void insert( SmartPointer* pointer, int deviceId );
      
      void remove( SmartPointer* pointer, int deviceId );
      
      bool synchronizeDevice( int deviceId );
      
   protected:
      
      typedef std::list< SmartPointer* > ListType;   
      
      std::vector< ListType > pointersOnDevices;
      
      int devicesCount;
};

