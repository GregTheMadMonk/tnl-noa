/***************************************************************************
                          SmartPointersRegister.h  -  description
                             -------------------
    begin                : Apr 29, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <unordered_set>
#include <unordered_map>
#include <TNL/Pointers/SmartPointer.h>
#include <TNL/Assert.h>

class SmartPointersRegister
{

   public:

      void insert( SmartPointer* pointer, int deviceId )
      {
         pointersOnDevices[ deviceId ].insert( pointer );
      }

      void remove( SmartPointer* pointer, int deviceId )
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

      bool synchronizeDevice( int deviceId )
      {
         try {
            const auto & set = pointersOnDevices.at( deviceId );
            for( auto&& it : set )
               ( *it ).synchronize();
            return true;
         }
         catch( const std::out_of_range& ) {
            return false;
         }
      }

   protected:

      typedef std::unordered_set< SmartPointer* > SetType;

      std::unordered_map< int, SetType > pointersOnDevices;
};
