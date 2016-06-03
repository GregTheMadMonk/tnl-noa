/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

/***************************************************************************
                          tnlSharedPointer.h  -  description
                             -------------------
    begin                : May 6, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#pragma once

#include <utility>
#include <core/tnlHost.h>
#include <core/tnlCuda.h>
#include <core/tnlSmartPointer.h>

template< typename Object, typename Device = typename Object::DeviceType >
class tnlSharedPointer
{  
};

template< typename Object >
class tnlSharedPointer< Object, tnlHost > : public tnlSmartPointer
{
   public:
      
      typedef Object ObjectType;
      typedef tnlHost DeviceType;
      typedef tnlSharedPointer< Object, tnlHost > ThisType;
         
      explicit  tnlSharedPointer()
      : counter( new int )
      {
         std::cerr << "Creating new shared pointer..." << std::endl;
         this->pointer = new Object();
         *( this->counter ) = 1;
      }
      
      template< typename... Args >
      explicit tnlSharedPointer( const Args... args )
      : counter( new int )
      {
         std::cerr << "Creating new shared pointer..." << std::endl;
         this->pointer = new Object( args... );
         *( this->counter ) = 1;
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
      
      template< typename Device = tnlHost >
      const Object& getData() const
      {
         return *( this->pointer );
      }

      Object& modifyData()
      {
         return *( this->pointer );
      }
      
      const ThisType& operator=( const ThisType& ptr )
      {
         this->free();
         this->pointer = ptr.pointer;
         this->counter = ptr.counter;
         *( this->counter )++;
         return *this;
      }
      
      const ThisType& operator=( ThisType&& ptr )
      {
         if( this-> pointer )
            delete this->pointer;
         this->pointer = ptr.pointer;
         ptr.pointer= NULL;
         this->counter = ptr.counter;
         ptr.counter = NULL;
         return *this;
      }
            
      bool synchronize()
      {
         return true;
      }
      
      ~tnlSharedPointer()
      {
         this->free();
      }

      
   protected:
      
      void free()
      {
         if( this->counter )
         {
            if( ! --*( this->counter ) )               
               delete this->pointer;
            std::cerr << "Deleting data..." << std::endl;
         }

      }
      
      Object* pointer;
      
      int* counter;
};

template< typename Object >
class tnlSharedPointer< Object, tnlCuda > : public tnlSmartPointer
{
   public:
      
      typedef Object ObjectType;
      typedef tnlHost DeviceType;
      typedef tnlSharedPointer< Object, tnlHost > ThisType;
         
      explicit tnlSharedPointer()
      : modified( false ),
        counter( new int )
      {
         std::cerr << "Creating new shared pointer..." << std::endl;
         this->pointer = new Object();
         *( this->counter )= 1;
#ifdef HAVE_CUDA         
         cudaMalloc( ( void** )  &this->cuda_pointer, sizeof( Object ) );
         cudaMemcpy( this->cuda_pointer, this->pointer, sizeof( Object ), cudaMemcpyHostToDevice );
         tnlCuda::insertSmartPointer( this );
#endif         
      }
      
      template< typename... Args >
      explicit tnlSharedPointer( const Args... args )
      : modified( false ),
        counter( new int )
      {
         std::cerr << "Creating new shared pointer..." << std::endl;
         this->pointer = new Object( args... );
         *( this->counter )= 1;
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
      
      template< typename Device = tnlHost >   
      __cuda_callable__
      const Object& getData() const
      {
         static_assert( std::is_same< Device, tnlHost >::value || std::is_same< Device, tnlCuda >::value, "Only tnlHost or tnlCuda devices are accepted here." );
         if( std::is_same< Device, tnlHost >::value )
            return *( this->pointer );
         if( std::is_same< Device, tnlCuda >::value )
            return *( this->cuda_pointer );            
      }

      Object& modifyData()
      {
         this->modified = true;
         return *( this->pointer );
      }
      
      const ThisType& operator=( ThisType&& ptr )
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
         this->counter = ptr.counter;
         ptr.pointer= NULL;
         ptr.cuda_pointer = NULL;
         ptr.modified = false;
         ptr.counter = NULL;
         return *this;
      }
      
      const ThisType& operator=( const ThisType& ptr )
      {
         this->free();
         this->pointer = ptr.pointer;
         this->cuda_pointer = ptr.cuda_pointer;
         this->modified = ptr.modified;
         this->counter = ptr.counter;
         *( this->counter )++;
         return *this;
      }
      
      
      bool synchronize()
      {
#ifdef HAVE_CUDA
         if( this-> modified )
         {
            std::cerr << "Synchronizing data..." << std::endl;
            cudaMemcpy( this->cuda_pointer, this->pointer, sizeof( Object ), cudaMemcpyHostToDevice );
            if( ! checkCudaDevice )
               return false;
            return true;
         }
#else         
         return false;
#endif         
      }
            
      ~tnlSharedPointer()
      {
         this->free();
#ifdef HAVE_CUDA         
         tnlCuda::removeSmartPointer( this );
#endif         
      }
      
   protected:
      
      void free()
      {
         if( this->counter )
         {
            if( ! --*( this->counter ) )
            {
               if( this->pointer )
                  delete this->pointer;
#ifdef HAVE_CUDA
               if( this->cuda_pointer )
                  cudaFree( this->cuda_pointer );
#endif         
               std::cerr << "Deleting data..." << std::endl;
            }
         }
         
      }
      
      Object *pointer, *cuda_pointer;
      
      bool modified;
      
      int* counter;
};


