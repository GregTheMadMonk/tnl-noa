/***************************************************************************
                          MIC.cpp  -  description
                             -------------------
    begin                : Feb 10, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */
#include <TNL/Devices/MIC.h>


namespace TNL {
namespace Devices {
	
SmartPointersRegister MIC::smartPointersRegister;
Timer MIC::smartPointersSynchronizationTimer;

void MIC::insertSmartPointer( SmartPointer* pointer )
{
   smartPointersRegister.insert( pointer, -1 );
}

void MIC::removeSmartPointer( SmartPointer* pointer )
{
   smartPointersRegister.remove( pointer, -1 );
}

bool MIC::synchronizeDevice( int deviceId )
{
   smartPointersSynchronizationTimer.start();
   bool b = smartPointersRegister.synchronizeDevice( deviceId );
   smartPointersSynchronizationTimer.stop();
   return b;
}


} // namespace Devices
} // namespace TNL