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
#include <TNL/Pointers/SmartPointersRegister.h>
#include <TNL/Devices/Cuda.h>

void SmartPointersRegister::insert( SmartPointer* pointer, int deviceId )
{
   //std::cerr << "Inserting pointer " << pointer << " to the register..." << std::endl;
   pointersOnDevices[ deviceId ].insert( pointer );
}

void SmartPointersRegister::remove( SmartPointer* pointer, int deviceId )
{
   try {
      pointersOnDevices.at( deviceId ).erase( pointer );
   }
   catch( const std::out_of_range& ) {
      std::cerr << "Given deviceId " << deviceId << " does not have any pointers yet. "
                << "Requested to remove pointer " << pointer << ". "
                << "This is most likely a bug in the smart pointer." << std::endl;
      throw;
   }
}

bool SmartPointersRegister::synchronizeDevice( int deviceId )
{
   try {
      const auto & set = pointersOnDevices.at( deviceId );
      for( auto&& it : set )
         ( *it ).synchronize();
      return TNL_CHECK_CUDA_DEVICE;
   }
   catch( const std::out_of_range& ) {
      return false;
   }
}
