/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

/***************************************************************************
                          SmartPointersRegister.cpp  -  description
                             -------------------
    begin                : Apr 29, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#include <iostream>
#include <TNL/SmartPointersRegister.h>

SmartPointersRegister::SmartPointersRegister( int devicesCount )
{
   Assert( devicesCount > 0, std::cerr << "devicesCount = " << devicesCount );
   pointersOnDevices.resize( devicesCount );
   this->devicesCount = devicesCount;
}

void SmartPointersRegister::insert( SmartPointer* pointer, int deviceId )
{
   Assert( deviceId >= 0 && deviceId < this->devicesCount,
              std::cerr << "deviceId = " << deviceId << " devicesCount = " << this->devicesCount );
   //std::cerr << "Inserting pointer " << pointer << " to the register..." << std::endl;
   pointersOnDevices[ deviceId ].push_back( pointer );
}

void SmartPointersRegister::remove( SmartPointer* pointer, int deviceId )
{
   Assert( deviceId >= 0 && deviceId < this->devicesCount,
              std::cerr << "deviceId = " << deviceId << " devicesCount = " << this->devicesCount );   
   pointersOnDevices[ deviceId ].remove( pointer );
}


bool SmartPointersRegister::synchronizeDevice( int deviceId )
{
   for( ListType::iterator it = pointersOnDevices[ deviceId ].begin();
        it != pointersOnDevices[ deviceId ].end();
        it++ )
      ( *it )->synchronize();            
}
