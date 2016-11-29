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

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/SmartPointer.h>

#include <cstring>


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
      
      operator bool()
      {
         return this->pointer;
      }

      template< typename Device = Devices::Host >
      const Object& getData() const
      {
         return *( this->pointer );
      }

      template< typename Device = Devices::Host >
      Object& modifyData()
      {
         return *( this->pointer );
      }
      
      const ThisType& operator=( ThisType& ptr )
      {
         if( this->pointer )
            delete this->pointer;
         this->pointer = ptr.pointer;
         ptr.pointer = nullptr;
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
      typedef Devices::Cuda DeviceType;
      typedef UniquePointer< Object, Devices::Cuda > ThisType;
         
      template< typename... Args >
      explicit  UniquePointer( const Args... args )
      : pd( nullptr ),
        cuda_pointer( nullptr )
      {
         this->allocate( args... );
      }
      
      const Object* operator->() const
      {
         return &this->pd->data;
      }
      
      Object* operator->()
      {
         this->pd->maybe_modified = true;
         return &this->pd->data;
      }
      
      const Object& operator *() const
      {
         return this->pd->data;
      }
      
      Object& operator *()
      {
         this->pd->maybe_modified = true;
         return this->pd->data;
      }
      
      operator bool()
      {
         return this->pd;
      }

      template< typename Device = Devices::Host >      
      const Object& getData() const
      {
         static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value, "Only Devices::Host or Devices::Cuda devices are accepted here." );
         TNL_ASSERT( this->pd, );
         TNL_ASSERT( this->cuda_pointer, );
         if( std::is_same< Device, Devices::Host >::value )
            return this->pd->data;
         if( std::is_same< Device, Devices::Cuda >::value )
            return *( this->cuda_pointer );            
      }

      template< typename Device = Devices::Host >
      Object& modifyData()
      {
         static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value, "Only Devices::Host or Devices::Cuda devices are accepted here." );
         TNL_ASSERT( this->pd, );
         TNL_ASSERT( this->cuda_pointer, );
         if( std::is_same< Device, Devices::Host >::value )
         {
            this->pd->maybe_modified = true;
            return this->pd->data;
         }
         if( std::is_same< Device, Devices::Cuda >::value )
            return *( this->cuda_pointer );
      }
      
      const ThisType& operator=( ThisType& ptr )
      {
         this->free();
         this->pd = ptr.pd;
         this->cuda_pointer = ptr.cuda_pointer;
         ptr.pd = nullptr;
         ptr.cuda_pointer = nullptr;
         return *this;
      }
      
      const ThisType& operator=( ThisType&& ptr )
      {
         return this->operator=( ptr );
      }      
      
      bool synchronize()
      {
         if( ! this->pd )
            return true;
#ifdef HAVE_CUDA
         if( this->modified() )
         {
            cudaMemcpy( (void*) this->cuda_pointer, (void*) &this->pd->data, sizeof( Object ), cudaMemcpyHostToDevice );
            if( ! checkCudaDevice )
               return false;
            this->set_last_sync_state();
            return true;
         }
         return true;
#else         
         return false;
#endif         
      }
            
      ~UniquePointer()
      {
         this->free();
         Devices::Cuda::removeSmartPointer( this );
      }
      
   protected:

      struct PointerData
      {
         Object data;
         char data_image[ sizeof(Object) ];
         bool maybe_modified;

         template< typename... Args >
         explicit PointerData( Args... args )
         : data( args... ),
           maybe_modified( false )
         {}
      };

      template< typename... Args >
      bool allocate( Args... args )
      {
         this->pd = new PointerData( args... );
         if( ! this->pd )
            return false;
         // pass to device
         this->cuda_pointer = Devices::Cuda::passToDevice( this->pd->data );
         if( ! this->cuda_pointer )
            return false;
         // set last-sync state
         this->set_last_sync_state();
         Devices::Cuda::insertSmartPointer( this );
         return true;
      }

      void set_last_sync_state()
      {
         TNL_ASSERT( this->pd, );
         std::memcpy( (void*) &this->pd->data_image, (void*) &this->pd->data, sizeof( ObjectType ) );
         this->pd->maybe_modified = false;
      }

      bool modified()
      {
         TNL_ASSERT( this->pd, );
         // optimization: skip bitwise comparison if we're sure that the data is the same
         if( ! this->pd->maybe_modified )
            return false;
         return std::memcmp( (void*) &this->pd->data_image, (void*) &this->pd->data, sizeof( ObjectType ) ) != 0;
      }

      void free()
      {
         if( this->pd )
            delete this->pd;
         if( this->cuda_pointer )
            Devices::Cuda::freeFromDevice( this->cuda_pointer );
      }
      
      PointerData* pd;

      // cuda_pointer can't be part of PointerData structure, since we would be
      // unable to dereference this-pd on the device
      Object* cuda_pointer;
};

} // namespace TNL

