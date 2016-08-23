/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

/***************************************************************************
                          UniquePointer.h  -  description
                             -------------------
    begin                : May 6, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#pragma once

#include <utility>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/SmartPointer.h>

namespace TNL { 

template< typename Object, typename Device = typename Object::DeviceType >
class UniquePointer
{  
};

template< typename Object >
class UniquePointer< Object, Devices::Host > : public SmartPointer
{
   public:
      
      typedef Object ObjectType;
      typedef Devices::Host DeviceType;
      typedef UniquePointer< Object, Devices::Host > ThisType;
         
      UniquePointer()
      {
         this->pointer = new Object();
      }
      
      template< typename... Args >
      UniquePointer( const Args... args )
      {
         this->pointer = new Object( args... );
      }
      
      const Object* operator->() const
      {
         return this->pointer;
      }
      
      Object* operator->()
      {
         return this->pointer;
      }
      
      const Object& operator *() const
      {
         return *( this->pointer );
      }
      
      Object& operator *()
      {
         return *( this->pointer );
      }
      
      template< typename Device = Devices::Host >
      const Object& getData() const
      {
         return *( this->pointer );
      }

      Object& modifyData()
      {
         return *( this->pointer );
      }
      
      const ThisType& operator=( ThisType& ptr )
      {
         if( this-> pointer )
            delete this->pointer;
         this->pointer = ptr.pointer;
         ptr.pointer= NULL;
         return *this;
      }
      
      const ThisType& operator=( ThisType&& ptr )
      {
         return this->operator=( ptr );         
      }      
      
      bool synchronize()
      {
         return true;
      }
      
      ~UniquePointer()
      {
         if( this->pointer )
            delete this->pointer;
      }

      
   protected:
      
      Object* pointer;
};

template< typename Object >
class UniquePointer< Object, Devices::Cuda > : public SmartPointer
{
   public:
      
      typedef Object ObjectType;
      typedef Devices::Host DeviceType;
      typedef UniquePointer< Object, Devices::Host > ThisType;
         
      UniquePointer()
      : modified( false )
      {
         this->pointer = new Object();
#ifdef HAVE_CUDA         
         cudaMalloc( ( void** )  &this->cuda_pointer, sizeof( Object ) );
         cudaMemcpy( this->cuda_pointer, this->pointer, sizeof( Object ), cudaMemcpyHostToDevice );
         tnlCuda::insertSmartPointer( this );
#endif         
      }
      
      template< typename... Args >
      UniquePointer( const Args... args )
      : modified( false )
      {
         this->pointer = new Object( args... );
#ifdef HAVE_CUDA         
         cudaMalloc( ( void** )  &this->cuda_pointer, sizeof( Object ) );
         cudaMemcpy( this->cuda_pointer, this->pointer, sizeof( Object ), cudaMemcpyHostToDevice );
         tnlCuda::insertSmartPointer( this );
#endif                  
      }
      
      const Object* operator->() const
      {
         return this->pointer;
      }
      
      Object* operator->()
      {
         this->modified = true;
         return this->pointer;
      }
      
      const Object& operator *() const
      {
         return *( this->pointer );
      }
      
      Object& operator *()
      {
         this->modified = true;
         return *( this->pointer );
      }
      
      template< typename Device = Devices::Host >      
      const Object& getData() const
      {
         static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value, "Only Devices::Host or tnlCuda devices are accepted here." );
         if( std::is_same< Device, Devices::Host >::value )
            return *( this->pointer );
         if( std::is_same< Device, Devices::Cuda >::value )
            return *( this->cuda_pointer );            
      }

      Object& modifyData()
      {
         this->modified = true;
         return *( this->pointer );
      }
      
      const ThisType& operator=( ThisType& ptr )
      {
         if( this-> pointer )
            delete this->pointer;
#ifdef HAVE_CUDA
         if( this->cuda_pointer )
            cudaFree( this->cuda_pointer );
#endif                  
         this->pointer = ptr.pointer;
         this->cuda_pointer = ptr.cuda_pointer;
         this->modified = ptr.modified;
         ptr.pointer= NULL;
         ptr.cuda_pointer = NULL;
         ptr.modified = false;
         return *this;
      }
      
      const ThisType& operator=( ThisType&& ptr )
      {
         return this->operator=( ptr );
      }      
      
      bool synchronize()
      {
#ifdef HAVE_CUDA
         if( this-> modified )
         {
            cudaMemcpy( this->cuda_pointer, this->pointer, sizeof( Object ), cudaMemcpyHostToDevice );
            if( ! checkCudaDevice )
               return false;
            return true;
         }
#else         
         return false;
#endif         
      }
            
      ~UniquePointer()
      {
         if( this->pointer )
            delete this->pointer;
#ifdef HAVE_CUDA
         if( this->cuda_pointer )
            cudaFree( this->cuda_pointer );
         tnlCuda::removeSmartPointer( this );
#endif         
      }
      
   protected:
      
      Object *pointer, cuda_pointer;
      
      bool modified;      
};

} // namespace TNL

