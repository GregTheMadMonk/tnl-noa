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
      UniquePointer( const Args... args )
      : pointer( 0 ), cuda_pointer( 0 ),
        last_sync_state( 0 )
      {
         this->pointer = new Object( args... );
         this->cuda_pointer = Devices::Cuda::passToDevice( *this->pointer );
         this->last_sync_state = ::operator new( sizeof( Object ) );
         this->set_last_sync_state();
         Devices::Cuda::insertSmartPointer( this );
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
         static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value, "Only Devices::Host or Devices::Cuda devices are accepted here." );
         if( std::is_same< Device, Devices::Host >::value )
            return *( this->pointer );
         if( std::is_same< Device, Devices::Cuda >::value )
            return *( this->cuda_pointer );            
      }

      template< typename Device = Devices::Host >
      Object& modifyData()
      {
         static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value, "Only Devices::Host or Devices::Cuda devices are accepted here." );
         if( std::is_same< Device, Devices::Host >::value )
         {
            return *( this->pointer );
         }
         if( std::is_same< Device, Devices::Cuda >::value )
         {
            return *( this->cuda_pointer );
         }
      }
      
      const ThisType& operator=( ThisType& ptr )
      {
         if( this->pointer )
            delete this->pointer;
         if( this->cuda_pointer )
            Devices::Cuda::freeFromDevice( this->cuda_pointer );
         this->pointer = ptr.pointer;
         this->cuda_pointer = ptr.cuda_pointer;
         this->last_sync_state = ptr.last_sync_state;
         ptr.pointer = nullptr;
         ptr.cuda_pointer = nullptr;
         ptr.last_sync_state = nullptr;
         return *this;
      }
      
      const ThisType& operator=( ThisType&& ptr )
      {
         return this->operator=( ptr );
      }      
      
      bool synchronize()
      {
#ifdef HAVE_CUDA
         if( this->modified() )
         {
            cudaMemcpy( (void*) this->cuda_pointer, (void*) this->pointer, sizeof( Object ), cudaMemcpyHostToDevice );
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
         if( this->pointer )
            delete this->pointer;
         if( this->cuda_pointer )
            Devices::Cuda::freeFromDevice( this->cuda_pointer );
         if( this->last_sync_state )
            ::operator delete( this->last_sync_state );
         Devices::Cuda::removeSmartPointer( this );
      }
      
   protected:

      void set_last_sync_state()
      {
         std::memcpy( (void*) this->last_sync_state, (void*) this->pointer, sizeof( ObjectType ) );
      }

      bool modified()
      {
         return std::memcmp( (void*) this->last_sync_state, (void*) this->pointer, sizeof( ObjectType ) ) != 0;
      }
      
      Object *pointer, *cuda_pointer;
      
      void* last_sync_state;
};

} // namespace TNL

