/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

/***************************************************************************
                          SharedPointer.h  -  description
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

/***
 * Use the lazy mode if you do not want to call the object constructor in the
 * shared pointer constructor. You may call it later via the method recreate.
 */
template< typename Object,
          typename Device = typename Object::DeviceType,
          bool lazy = false,
          bool isConst = std::is_const< Object >::value >
class SharedPointer
{
   static_assert( ! std::is_same< Device, void >::value, "The device cannot be void. You need to specify the device explicitly in your code." );
};

/****
 * Non-const specialization
 */
template< typename Object, bool lazy >
class SharedPointer< Object, Devices::Host, lazy, false > : public SmartPointer
{   
   public:
      
      typedef Object ObjectType;
      typedef Devices::Host DeviceType;
      typedef SharedPointer< Object, Devices::Host, lazy, false > ThisType;
      typedef SharedPointer< const Object, Devices::Host, lazy, true > ConstThisType;
         
      template< typename... Args >
      explicit  SharedPointer( Args... args )
      : pointer( 0 ), counter( 0 )
      {
         if( ! lazy )
         {
            this->counter = new int;
            this->pointer = new Object( args... );
            *( this->counter ) = 1;
         }
      }
      
      SharedPointer( const ThisType& pointer )
      : pointer( pointer.pointer ),
        counter( pointer.counter )
      {
         *counter += 1;
      }
      
      template< typename... Args >
      bool recreate( Args... args )
      {         
         std::cerr << "Creating new shared pointer..." << std::endl;
         if( ! this->counter )
         {
            this->counter = new int;
            *this->counter = 1;
            this->pointer = new ObjectType( args... );
            return true;
         }
         if( *this->counter == 1 )
         {
            /****
             * The object is not shared
             */
            this->pointer->~ObjectType();
            new ( this->pointer ) ObjectType( args... );
            return true;
         }
         ( *this->counter )--;
         this->pointer = new Object( args... );
         this->counter = new int;
         if( ! this->pointer || ! this->counter )
            return false;
         *( this->counter ) = 1;
         return true;         
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

      template< typename Device = Devices::Host >
      Object& modifyData()
      {
         return *( this->pointer );
      }
      
      const ThisType& operator=( const ThisType& ptr )
      {
         this->free();
         this->pointer = ptr.pointer;
         this->counter = ptr.counter;
         *( this->counter ) += 1;
         return *this;
      }      
      
      const ThisType& operator=( const ThisType&& ptr )
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
      
      ~SharedPointer()
      {
         this->free();
      }

      
   protected:
      
      void free()
      {
         if( ! this->pointer )
            return;
         if( this->counter )
         {
            if( ! --*( this->counter ) )
            {
               delete this->pointer;
               //std::cerr << "Deleting data..." << std::endl;
            }
         }

      }
      
      Object* pointer;
      
      int* counter;
};

/****
 * Const specialization
 */
template< typename Object, bool lazy >
class SharedPointer< Object, Devices::Host, lazy, true > : public SmartPointer
{   
   public:
      
      typedef Object ObjectType;
      typedef Devices::Host DeviceType;
      typedef SharedPointer< Object, Devices::Host, lazy, true > ThisType;
      typedef typename std::remove_const< Object >::type NonConstObjectType;
         
      template< typename... Args >
      explicit  SharedPointer( Args... args )
      : counter( 0 ), pointer( 0 )
      {
         if( ! lazy )
         {
            this->counter = new int;
            this->pointer = new Object( args... );
            *( this->counter ) = 1;
         }
      }
      
      SharedPointer( const ThisType& pointer )
      : pointer( pointer.pointer ),
        counter( pointer.counter )
      {
         *counter += 1;
      }
      
      SharedPointer( const SharedPointer< NonConstObjectType, Devices::Host, lazy >& pointer )
      : pointer( pointer.pointer ),
        counter( pointer.counter )
      {
         *counter += 1;
      }

      
      template< typename... Args >
      bool recreate( Args... args )
      {         
         std::cerr << "Creating new shared pointer..." << std::endl;
         if( ! this->counter )
         {
            this->counter = new int;
            *this->counter = 1;
            this->pointer = new ObjectType( args... );
            return true;
         }
         if( *this->counter == 1 )
         {
            /****
             * The object is not shared
             */
            this->pointer->~ObjectType();
            new ( this->pointer ) ObjectType( args... );
            return true;
         }
         ( *this->counter )--;
         this->pointer = new Object( args... );
         this->counter = new int;
         if( ! this->pointer || ! this->counter )
            return false;
         *( this->counter ) = 1;
         return true;         
      }      
      
      const Object* operator->() const
      {
         return this->pointer;
      }
            
      const Object& operator *() const
      {
         return *( this->pointer );
      }
      
      template< typename Device = Devices::Host >
      const Object& getData() const
      {
         return *( this->pointer );
      }
      
      const ThisType& operator=( const SharedPointer< NonConstObjectType, Devices::Host >& ptr )
      {
         this->free();
         this->pointer = ptr.pointer;
         this->counter = ptr.counter;
         *( this->counter ) += 1;
         return *this;
      }      

      
      const ThisType& operator=( const ThisType& ptr )
      {
         this->free();
         this->pointer = ptr.pointer;
         this->counter = ptr.counter;
         *( this->counter ) += 1;
         return *this;
      }      
      
      const ThisType& operator=( const ThisType&& ptr )
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
      
      ~SharedPointer()
      {
         this->free();
      }

      
   protected:
      
      void free()
      {
         if( ! this->pointer )
            return;
         if( this->counter )
         {
            if( ! --*( this->counter ) )
            {
               delete this->pointer;
               //std::cerr << "Deleting data..." << std::endl;
            }
         }

      }
      
      const Object* pointer;
      
      int* counter;
};

/****
 * Non-const specialization for CUDA
 */
template< typename Object, bool lazy >
class SharedPointer< Object, Devices::Cuda, lazy, false > : public SmartPointer
{
   public:
      
      typedef Object ObjectType;
      typedef Devices::Host DeviceType;
      typedef SharedPointer< Object, Devices::Cuda, lazy > ThisType;

      template< typename... Args >
      explicit  SharedPointer( Args... args )
      : counter( 0 ), cuda_pointer( 0 ), 
        pointer( 0 ), modified( false )
      {
         if( ! lazy )
         {
            this->counter = new int;
            this->pointer = new Object( args... );
#ifdef HAVE_CUDA         
            this->cuda_pointer = Devices::Cuda::passToDevice( *this->pointer );
            if( ! checkCudaDevice )
               return;
            Devices::Cuda::insertSmartPointer( this );
#endif            
         }
      }
                  
      SharedPointer( const ThisType& pointer )
      : pointer( pointer.pointer ),
        cuda_pointer( pointer.cuda_pointer ),
        counter( pointer.counter ),
        modified( pointer.modified )
      {
         *counter += 1;
      }

      template< typename... Args >
      bool recreate( Args... args )
      {
         //std::cerr << "Creating new shared pointer..." << std::endl;
         if( ! this->counter )
         {
            this->counter = new int;
            *this->counter = 1;
            this->pointer = new ObjectType( args... );
#ifdef HAVE_CUDA         
            this->cuda_pointer = Devices::Cuda::passToDevice( *this->object );
            if( ! checkCudaDevice )
               return false;
            Devices::Cuda::insertSmartPointer( this );
#endif                 
            return true;
         }
         if( *this->counter == 1 )
         {
            /****
             * The object is not shared
             */
            this->pointer->~ObjectType();
            new ( this->pointer ) ObjectType( args... );
#ifdef HAVE_CUDA                     
            cudaMemcpy( this->cuda_pointer, this->pointer, sizeof( Object ), cudaMemcpyHostToDevice );
#endif            
            return true;
         }

         this->modified = false;
         this->counter= new int;
         this->pointer = new Object( args... );
         if( ! this->pointer || ! this->counter )
            return false;
         *( this->counter )= 1;         
#ifdef HAVE_CUDA         
         cudaMalloc( ( void** )  &this->cuda_pointer, sizeof( Object ) );
         cudaMemcpy( this->cuda_pointer, this->pointer, sizeof( Object ), cudaMemcpyHostToDevice );
         if( ! checkCudaDevice )
            return false;
         Devices::Cuda::insertSmartPointer( this );
#endif
         return true;
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
      __cuda_callable__
      const Object& getData() const
      {
         static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value, "Only Devices::Host or tnlCuda devices are accepted here." );
         Assert( this->pointer, );
         Assert( this->cuda_pointer, );
         if( std::is_same< Device, Devices::Host >::value )
            return *( this->pointer );
         if( std::is_same< Device, Devices::Cuda >::value )
            return *( this->cuda_pointer );            
      }

      template< typename Device = Devices::Host >
      __cuda_callable__
      Object& modifyData()
      {
         if( std::is_same< Device, Devices::Host >::value )
         {
            this->modified = true;
            return *( this->pointer );
         }
         if( std::is_same< Device, Devices::Cuda >::value )
         {
            return *( this->cuda_pointer );            
         }
      }
      
      /*const ThisType& operator=( ThisType&& ptr )
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
      }*/
      
      const ThisType& operator=( const ThisType& ptr )
      {
         this->free();
         this->pointer = ptr.pointer;
         this->cuda_pointer = ptr.cuda_pointer;
         this->modified = ptr.modified;
         this->counter = ptr.counter;
         *( this->counter ) += 1;
         return *this;
      }
      
      
      bool synchronize()
      {
#ifdef HAVE_CUDA
         if( this->modified )
         {
            //std::cerr << "Synchronizing data ( " << sizeof( ObjectType ) << " bytes ) to adress " << this->cuda_pointer << "..." << std::endl;
            Assert( this->pointer, );
            Assert( this->cuda_pointer, );
            cudaMemcpy( this->cuda_pointer, this->pointer, sizeof( ObjectType ), cudaMemcpyHostToDevice );            
            if( ! checkCudaDevice )
               return false;
            this->modified = false;
            return true;
         }
#else         
         return false;
#endif         
      }
            
      ~SharedPointer()
      {
         this->free();
#ifdef HAVE_CUDA         
         Devices::Cuda::removeSmartPointer( this );
#endif         
      }
      
   protected:
      
      void free()
      {
         if( ! this->pointer )
            return;
         if( this->counter )
         {
            if( ! --*( this->counter ) )
            {
               if( this->pointer )
                  delete this->pointer;
#ifdef HAVE_CUDA
               if( this->cuda_pointer )
                  cudaFree( this->cuda_pointer );
               checkCudaDevice;
#endif         
               //std::cerr << "Deleting data..." << std::endl;
            }
         }
         
      }
      
      Object *pointer, *cuda_pointer;
      
      bool modified;
      
      int* counter;
};


/****
 * Const specialization for CUDA
 */
template< typename Object, bool lazy >
class SharedPointer< Object, Devices::Cuda, lazy, true > : public SmartPointer
{
   public:
      
      typedef Object ObjectType;
      typedef Devices::Host DeviceType;
      typedef SharedPointer< Object, Devices::Cuda, lazy > ThisType;
      typedef typename std::remove_const< Object >::type NonConstObjectType;      

      template< typename... Args >
      explicit  SharedPointer( Args... args )
      : counter( 0 ), cuda_pointer( 0 ), 
        pointer( 0 ), modified( false )
      {
         if( ! lazy )
         {
            this->counter = new int;
            this->pointer = new Object( args... );
#ifdef HAVE_CUDA         
            this->cuda_pointer = Devices::Cuda::passToDevice( *this->pointer );
            if( ! checkCudaDevice )
               return;
            Devices::Cuda::insertSmartPointer( this );
#endif            
         }
      }
                  
      SharedPointer( const ThisType& pointer )
      : pointer( pointer.pointer ),
        cuda_pointer( pointer.cuda_pointer ),
        counter( pointer.counter ),
        modified( pointer.modified )
      {
         *counter += 1;
      }
      
      SharedPointer( const SharedPointer< NonConstObjectType, Devices::Cuda, lazy >& pointer )
      : pointer( pointer.pointer ),
        cuda_pointer( pointer.cuda_pointer ),
        counter( pointer.counter )
      {
         *counter += 1;
      }      

      template< typename... Args >
      bool recreate( Args... args )
      {
         std::cerr << "Creating new shared pointer..." << std::endl;
         if( ! this->counter )
         {
            this->counter = new int;
            *this->counter = 1;
            this->pointer = new ObjectType( args... );
#ifdef HAVE_CUDA         
            this->cuda_pointer = Devices::Cuda::passToDevice( *this->object );
            if( ! checkCudaDevice )
               return false;
            Devices::Cuda::insertSmartPointer( this );
#endif                 
            return true;
         }
         if( *this->counter == 1 )
         {
            /****
             * The object is not shared
             */
            this->pointer->~ObjectType();
            new ( this->pointer ) ObjectType( args... );
#ifdef HAVE_CUDA                     
            cudaMemcpy( this->cuda_pointer, this->pointer, sizeof( Object ), cudaMemcpyHostToDevice );
#endif            
            return true;
         }

         this->modified = false;
         this->counter= new int;
         this->pointer = new Object( args... );
         if( ! this->pointer || ! this->counter )
            return false;
         *( this->counter )= 1;         
#ifdef HAVE_CUDA         
         cudaMalloc( ( void** )  &this->cuda_pointer, sizeof( Object ) );
         cudaMemcpy( this->cuda_pointer, this->pointer, sizeof( Object ), cudaMemcpyHostToDevice );
         if( ! checkCudaDevice )
            return false;
         Devices::Cuda::insertSmartPointer( this );
#endif
         return true;
      }
      
      const Object* operator->() const
      {
         return this->pointer;
      }
      
      const Object& operator *() const
      {
         return *( this->pointer );
      }
      
      template< typename Device = Devices::Host >   
      __cuda_callable__
      const Object& getData() const
      {
         static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value, "Only Devices::Host or tnlCuda devices are accepted here." );
         Assert( this->pointer, );
         Assert( this->cuda_pointer, );
         if( std::is_same< Device, Devices::Host >::value )
            return *( this->pointer );
         if( std::is_same< Device, Devices::Cuda >::value )
            return *( this->cuda_pointer );            
      }

      
      /*const ThisType& operator=( ThisType&& ptr )
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
      }*/

      const ThisType& operator=( const SharedPointer< NonConstObjectType, Devices::Cuda >& ptr )
      {
         this->free();
         this->pointer = ptr.pointer;
         this->counter = ptr.counter;
         this->cuda_pointer = ptr.cuda_pointer;
         this->modified = ptr.modified;
         *( this->counter ) += 1;
         return *this;
      }      
      
      const ThisType& operator=( const ThisType& ptr )
      {
         this->free();
         this->pointer = ptr.pointer;
         this->cuda_pointer = ptr.cuda_pointer;
         this->modified = ptr.modified;
         this->counter = ptr.counter;
         *( this->counter ) += 1;
         return *this;
      }      
      
      bool synchronize()
      {
         return true;
      }
            
      ~SharedPointer()
      {
         this->free();
#ifdef HAVE_CUDA         
         Devices::Cuda::removeSmartPointer( this );
#endif         
      }
      
   protected:
      
      void free()
      {
         if( ! this->pointer )
            return;
         if( this->counter )
         {
            if( ! --*( this->counter ) )
            {
               if( this->pointer )
                  delete this->pointer;
#ifdef HAVE_CUDA
               if( this->cuda_pointer )
                  cudaFree( this->cuda_pointer );
               checkCudaDevice;
#endif         
               std::cerr << "Deleting data..." << std::endl;
            }
         }
         
      }
      
      Object *pointer, *cuda_pointer;
      
      bool modified;
      
      int* counter;
};

} // namespace TNL
