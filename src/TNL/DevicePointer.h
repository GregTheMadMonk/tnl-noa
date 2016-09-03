/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

/***************************************************************************
                          DevicePointer.h  -  description
                             -------------------
    begin                : Sep 1, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/SmartPointer.h>


namespace TNL {

/***
 * The DevicePointer is like SharedPointer, except it takes an existing host
 * object - there is no call to the ObjectType's constructor nor destructor.
 */
template< typename Object,
          typename Device = typename Object::DeviceType >
class DevicePointer
{
   static_assert( ! std::is_same< Device, void >::value, "The device cannot be void. You need to specify the device explicitly in your code." );
};

/****
 * Specialization for Devices::Host
 */
template< typename Object >
class DevicePointer< Object, Devices::Host > : public SmartPointer
{
   private:
      // Convenient template alias for controlling the selection of copy- and
      // move-constructors and assignment operators using SFINAE.
      // The type Object_ is "enabled" iff Object_ and Object are not the same,
      // but after removing const and volatile qualifiers they are the same.
      template< typename Object_ >
      using Enabler = std::enable_if< ! std::is_same< Object_, Object >::value &&
                                      std::is_same< typename std::remove_cv< Object >::type, Object_ >::value >;

      // friend class will be needed for templated assignment operators
      template< typename Object_, typename Device_ >
      friend class DevicePointer;

   public:

      typedef Object ObjectType;
      typedef Devices::Host DeviceType;
      typedef DevicePointer< Object, Devices::Host > ThisType;

      explicit  DevicePointer( ObjectType& obj )
      : pointer( nullptr )
      {
         this->pointer = &obj;
      }

      // this is needed only to avoid the default compiler-generated constructor
      DevicePointer( const ThisType& pointer )
      : pointer( pointer.pointer )
      {
      }

      // conditional constructor for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      DevicePointer( const DevicePointer< Object_, DeviceType >& pointer )
      : pointer( pointer.pointer )
      {
      }

      // this is needed only to avoid the default compiler-generated constructor
      DevicePointer( ThisType&& pointer )
      : pointer( pointer.pointer )
      {
         pointer.pointer = nullptr;
      }

      // conditional constructor for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      DevicePointer( DevicePointer< Object_, DeviceType >&& pointer )
      : pointer( pointer.pointer )
      {
         pointer.pointer = nullptr;
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
      __cuda_callable__
      const Object& getData() const
      {
         return *( this->pointer );
      }

      template< typename Device = Devices::Host >
      __cuda_callable__
      Object& modifyData()
      {
         return *( this->pointer );
      }

      // this is needed only to avoid the default compiler-generated operator
      const ThisType& operator=( const ThisType& ptr )
      {
         this->pointer = ptr.pointer;
         return *this;
      }

      // conditional operator for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      const ThisType& operator=( const DevicePointer< Object_, DeviceType >& ptr )
      {
         this->pointer = ptr.pointer;
         return *this;
      }

      // this is needed only to avoid the default compiler-generated operator
      const ThisType& operator=( ThisType&& ptr )
      {
         this->pointer = ptr.pointer;
         ptr.pointer = nullptr;
         return *this;
      }

      // conditional operator for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      const ThisType& operator=( DevicePointer< Object_, DeviceType >&& ptr )
      {
         this->pointer = ptr.pointer;
         ptr.pointer = nullptr;
         return *this;
      }

      bool synchronize()
      {
         return true;
      }

      ~DevicePointer()
      {
      }


   protected:

      Object* pointer;
};

/****
 * Specialization for CUDA
 */
template< typename Object >
class DevicePointer< Object, Devices::Cuda > : public SmartPointer
{
   private:
      // Convenient template alias for controlling the selection of copy- and
      // move-constructors and assignment operators using SFINAE.
      // The type Object_ is "enabled" iff Object_ and Object are not the same,
      // but after removing const and volatile qualifiers they are the same.
      template< typename Object_ >
      using Enabler = std::enable_if< ! std::is_same< Object_, Object >::value &&
                                      std::is_same< typename std::remove_cv< Object >::type, Object_ >::value >;

      // friend class will be needed for templated assignment operators
      template< typename Object_, typename Device_ >
      friend class DevicePointer;

   public:

      typedef Object ObjectType;
      typedef Devices::Cuda DeviceType;
      typedef DevicePointer< Object, Devices::Cuda > ThisType;

      explicit  DevicePointer( ObjectType& obj )
      : counter( 0 ), cuda_pointer( 0 ),
        pointer( 0 ), modified( false )
      {
         this->counter = new int( 1 );
         this->pointer = &obj;
         this->cuda_pointer = Devices::Cuda::passToDevice( *this->pointer );
         if( ! this->cuda_pointer )
            return;
         Devices::Cuda::insertSmartPointer( this );
      }

      // this is needed only to avoid the default compiler-generated constructor
      DevicePointer( const ThisType& pointer )
      : pointer( pointer.pointer ),
        cuda_pointer( pointer.cuda_pointer ),
        counter( pointer.counter ),
        modified( pointer.modified )
      {
         *counter += 1;
      }

      // conditional constructor for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      DevicePointer( const DevicePointer< Object_, DeviceType >& pointer )
      : pointer( pointer.pointer ),
        cuda_pointer( pointer.cuda_pointer ),
        counter( pointer.counter ),
        modified( pointer.modified )
      {
         *counter += 1;
      }

      // this is needed only to avoid the default compiler-generated constructor
      DevicePointer( ThisType&& pointer )
      : pointer( pointer.pointer ),
        cuda_pointer( pointer.cuda_pointer ),
        counter( pointer.counter ),
        modified( pointer.modified )
      {
         pointer.pointer = nullptr;
         pointer.cuda_pointer = nullptr;
         pointer.modified = false;
         pointer.counter = nullptr;
      }

      // conditional constructor for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      DevicePointer( DevicePointer< Object_, DeviceType >&& pointer )
      : pointer( pointer.pointer ),
        cuda_pointer( pointer.cuda_pointer ),
        counter( pointer.counter ),
        modified( pointer.modified )
      {
         pointer.pointer = nullptr;
         pointer.cuda_pointer = nullptr;
         pointer.modified = false;
         pointer.counter = nullptr;
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
         static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value, "Only Devices::Host or Devices::Cuda devices are accepted here." );
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
         static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value, "Only Devices::Host or Devices::Cuda devices are accepted here." );
         Assert( this->pointer, );
         Assert( this->cuda_pointer, );
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

      // this is needed only to avoid the default compiler-generated operator
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

      // conditional operator for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      const ThisType& operator=( const DevicePointer< Object_, DeviceType >& ptr )
      {
         this->free();
         this->pointer = ptr.pointer;
         this->cuda_pointer = ptr.cuda_pointer;
         this->modified = ptr.modified;
         this->counter = ptr.counter;
         *( this->counter ) += 1;
         return *this;
      }

      // this is needed only to avoid the default compiler-generated operator
      const ThisType& operator=( ThisType&& ptr )
      {
         this->free();
         this->pointer = ptr.pointer;
         this->cuda_pointer = ptr.cuda_pointer;
         this->modified = ptr.modified;
         this->counter = ptr.counter;
         ptr.pointer = nullptr;
         ptr.cuda_pointer = nullptr;
         ptr.modified = false;
         ptr.counter = nullptr;
         return *this;
      }

      // conditional operator for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      const ThisType& operator=( DevicePointer< Object_, DeviceType >&& ptr )
      {
         this->free();
         this->pointer = ptr.pointer;
         this->cuda_pointer = ptr.cuda_pointer;
         this->modified = ptr.modified;
         this->counter = ptr.counter;
         ptr.pointer = nullptr;
         ptr.cuda_pointer = nullptr;
         ptr.modified = false;
         ptr.counter = nullptr;
         return *this;
      }

      bool synchronize()
      {
#ifdef HAVE_CUDA
         if( this->modified )
         {
            Assert( this->pointer, );
            Assert( this->cuda_pointer, );
            cudaMemcpy( (void*) this->cuda_pointer, (void*) this->pointer, sizeof( ObjectType ), cudaMemcpyHostToDevice );
            if( ! checkCudaDevice ) {
               return false;
            }
            this->modified = false;
            return true;
         }
         return true;
#else
         return false;
#endif
      }

      ~DevicePointer()
      {
         this->free();
         Devices::Cuda::removeSmartPointer( this );
      }

   protected:

      void free()
      {
         if( this->counter )
         {
            if( ! --*( this->counter ) )
            {
               delete this->counter;
               this->counter = nullptr;
               if( this->cuda_pointer )
                  Devices::Cuda::freeFromDevice( this->cuda_pointer );
            }
         }

      }

      Object *pointer, *cuda_pointer;

      bool modified;

      int* counter;
};

} // namespace TNL
