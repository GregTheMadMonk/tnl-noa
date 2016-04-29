/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

/***************************************************************************
                          tnlDeviceObject.h  -  description
                             -------------------
    begin                : Apr 29, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#pragma once

#include <core/tnlHost.h>
#include <core/tnlCuda.h>
#include <core/tnlDeviceObjectBase.h>

template< typename Object,
          typename Device >
class tnlDeviceObject
{   
};

template< typename Object >
class tnlDeviceObject< Object, tnlHost > : public tnlDeviceObjectBase
{   
   public:
      
      tnlDeviceObject( Object& object )
      {
         this->pointer = &object;
      }
      
      Object* operator->()
      {
         return this->pointer;
      }
      
      const Object* operator->() const
      {
         return this->pointer;
      }
      
      const Object& get() const
      {
         return *this->pointer;
      }
      
      Object& modify()
      {
         return *this->pointer;
      }
      
      Object* getDevicePointer()
      {
         return this->pointer;
      }
      
      const Object* getDevicePointer() const
      {
         return this->pointer;
      }
      
      bool synchronize()
      {
         return true;
      }
            
   protected:
      
      Object* pointer;
};

template< typename Object >
class tnlDeviceObject< Object, tnlCuda > : public tnlDeviceObjectBase
{   
   public:
      
      tnlDeviceObject( Object& object )
      {
         this->host_pointer = &object;
#ifdef HAVE_CUDA
         cudaMalloc( ( void** ) &this->device_pointer, sizeof( Object ) );
         deviceId = tnlCuda::getDeviceId();
         tnlCuda::getDeviceObjectsContainer().enregister( this );
#endif         
      }

      
      Object* operator->()
      {
         return host_pointer;
      }
      
      const Object* operator->() const
      {
         return host_pointer;
      }
      
      const Object& get() const
      {
         return *host_pointer;
      }
      
      Object& modify()
      {
         return *host_pointer;
      }
      
      Object* getDevicePointer()
      {
         return device_pointer;
      }
      
      const Object* getDevicePointer() const
      {
         return device_pointer;
      }

      bool synchronize()
      {
         
      }
      
   protected:
      
      Object *host_pointer, *device_pointer;
      
      int deviceId;
};

