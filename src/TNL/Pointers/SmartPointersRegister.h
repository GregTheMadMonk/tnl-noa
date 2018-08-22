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

#include <unordered_set>
#include <unordered_map>
#include <TNL/Pointers/SmartPointer.h>
#include <TNL/Assert.h>

class SmartPointersRegister
{   
  
   public:
   
      void insert( SmartPointer* pointer, int deviceId );
      
      void remove( SmartPointer* pointer, int deviceId );
      
      bool synchronizeDevice( int deviceId );
      
   protected:
      
      typedef std::unordered_set< SmartPointer* > SetType;   
      
      std::unordered_map< int, SetType > pointersOnDevices;
};

