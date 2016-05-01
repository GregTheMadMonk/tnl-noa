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
      
      tnlDeviceObject()
      : modified( false ), device_pointer( 0 )
      {
#ifdef HAVE_CUDA         
         tnlCuda::getDeviceObjectsContainer().push( this );
#endif         
      }
      
      tnlDeviceObject( Object& object )
      : modified( true ), device_pointer( 0 )
      {
#ifdef HAVE_CUDA         
         tnlCuda::getDeviceObjectsContainer().push( this );         
#endif         
      }

      bool setObject( Object& object )
      {
#ifdef HAVE_CUDA
         this->host_pointer = &object;
         if( this->device_pointer )
            cudaFree( this->device_pointer );
         cudaMalloc( ( void** ) &this->device_pointer, sizeof( Object ) );
         if( ! checkCudaDevice )
            return false;
         deviceId = tnlCuda::getDeviceId();         
         this->modified = true;
         return true;
#else
         return false
#endif                  
      }
      
      template< typename Device >
      const Object& object() const
      {
         tnlAssert( std::is_same< Device, tnlHost >::value ||
                    std::is_same< Device, tnlCuda >::value, );

         if( std::is_same< Device, tnlHost >::value )
            return *this->pointer;
         return *this->device_pointer;
      }            
      
      Object& change()
      {
         this->modified = true;
         return *this->pointer;
      }            
            
      bool synchronize()
      {
#ifdef HAVE_CUDA
         if( this->modified )
         {
            cudaMemcpy( device_pointer, host_pointer, sizeof( Object ), cudaMemcpyHostToDevice );
            return checkCudaDevice;
         }
#else
         return false;
#endif                  
      }
      
      ~tnlDeviceObject()
      {
#ifdef HAVE_CUDA
         tnlCuda::getDeviceObjectsContainer().pull( this );
         if( this->device_pointer )
            cudaFree( this->device_pointer );
#endif         
      }
      
   protected:
      
      Object *host_pointer, *device_pointer;
      
      int deviceId;
      
      bool modified;
};

