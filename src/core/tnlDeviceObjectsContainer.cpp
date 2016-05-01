/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

/***************************************************************************
                          tnlDeviceObjectsContainer.cpp  -  description
                             -------------------
    begin                : Apr 29, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#include <core/tnlDeviceObjectsContainer.h>

tnlDeviceObjectsContainer::tnlDeviceObjectsContainer( int devicesCount )
{
   tnlAssert( devicesCount > 0, std::cerr << "devicesCount = " << devicesCount );
   objectsOnDevices.resize( devicesCount );
   this->devicesCount = devicesCount;
}

void tnlDeviceObjectsContainer::push( tnlDeviceObjectBase* object, int deviceId )
{
   tnlAssert( deviceId >= 0 && deviceId < this->devicesCount,
              std::cerr << "deviceId = " << deviceId << " devicesCount = " << this->devicesCount );
   objectsOnDevices[ deviceId ].push_back( object );
}

bool tnlDeviceObjectsContainer::synchronize()
{
   for( int i = 0; i < this->objectsOnDevices.size(); i++ )
      for( ListType::iterator it = objectsOnDevices[ i ].begin();
           it != objectsOnDevices[ i ].end();
           it++ )
         ( *it )->synchronize();            

}
