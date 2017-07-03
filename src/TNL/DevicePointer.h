/***************************************************************************
                          DevicePointer.h  -  description
                             -------------------
    begin                : Sep 1, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/SmartPointer.h>

#include <cstring>


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

      __cuda_callable__
      operator bool() const
      {
         return this->pointer;
      }

      __cuda_callable__
      bool operator!() const
      {
         return ! this->pointer;
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
      : pointer( nullptr ),
        pd( nullptr ),
        cuda_pointer( nullptr )
      {
         this->allocate( obj );
      }

      // this is needed only to avoid the default compiler-generated constructor
      DevicePointer( const ThisType& pointer )
      : pointer( pointer.pointer ),
        pd( (PointerData*) pointer.pd ),
        cuda_pointer( pointer.cuda_pointer )
      {
         this->pd->counter += 1;
      }

      // conditional constructor for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      DevicePointer( const DevicePointer< Object_, DeviceType >& pointer )
      : pointer( pointer.pointer ),
        pd( (PointerData*) pointer.pd ),
        cuda_pointer( pointer.cuda_pointer )
      {
         this->pd->counter += 1;
      }

      // this is needed only to avoid the default compiler-generated constructor
      DevicePointer( ThisType&& pointer )
      : pointer( pointer.pointer ),
        pd( (PointerData*) pointer.pd ),
        cuda_pointer( pointer.cuda_pointer )
      {
         pointer.pointer = nullptr;
         pointer.pd = nullptr;
         pointer.cuda_pointer = nullptr;
      }

      // conditional constructor for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      DevicePointer( DevicePointer< Object_, DeviceType >&& pointer )
      : pointer( pointer.pointer ),
        pd( (PointerData*) pointer.pd ),
        cuda_pointer( pointer.cuda_pointer )
      {
         pointer.pointer = nullptr;
         pointer.pd = nullptr;
         pointer.cuda_pointer = nullptr;
      }

      const Object* operator->() const
      {
         return this->pointer;
      }

      Object* operator->()
      {
         this->pd->maybe_modified = true;
         return this->pointer;
      }

      const Object& operator *() const
      {
         return *( this->pointer );
      }

      Object& operator *()
      {
         this->pd->maybe_modified = true;
         return *( this->pointer );
      }

      __cuda_callable__
      operator bool() const
      {
         return this->pd;
      }

      __cuda_callable__
      bool operator!() const
      {
         return ! this->pd;
      }

      template< typename Device = Devices::Host >
      __cuda_callable__
      const Object& getData() const
      {
         static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value, "Only Devices::Host or Devices::Cuda devices are accepted here." );
         TNL_ASSERT( this->pointer, );
         TNL_ASSERT( this->pd, );
         TNL_ASSERT( this->cuda_pointer, );
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
         TNL_ASSERT( this->pointer, );
         TNL_ASSERT( this->pd, );
         TNL_ASSERT( this->cuda_pointer, );
         if( std::is_same< Device, Devices::Host >::value )
         {
            this->pd->maybe_modified = true;
            return *( this->pointer );
         }
         if( std::is_same< Device, Devices::Cuda >::value )
            return *( this->cuda_pointer );
      }

      // this is needed only to avoid the default compiler-generated operator
      const ThisType& operator=( const ThisType& ptr )
      {
         this->free();
         this->pointer = ptr.pointer;
         this->pd = (PointerData*) ptr.pd;
         this->cuda_pointer = ptr.cuda_pointer;
         this->pd->counter += 1;
         return *this;
      }

      // conditional operator for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      const ThisType& operator=( const DevicePointer< Object_, DeviceType >& ptr )
      {
         this->free();
         this->pointer = ptr.pointer;
         this->pd = (PointerData*) ptr.pd;
         this->cuda_pointer = ptr.cuda_pointer;
         this->pd->counter += 1;
         return *this;
      }

      // this is needed only to avoid the default compiler-generated operator
      const ThisType& operator=( ThisType&& ptr )
      {
         this->free();
         this->pointer = ptr.pointer;
         this->pd = (PointerData*) ptr.pd;
         this->cuda_pointer = ptr.cuda_pointer;
         ptr.pointer = nullptr;
         ptr.pd = nullptr;
         ptr.cuda_pointer = nullptr;
         return *this;
      }

      // conditional operator for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      const ThisType& operator=( DevicePointer< Object_, DeviceType >&& ptr )
      {
         this->free();
         this->pointer = ptr.pointer;
         this->pd = (PointerData*) ptr.pd;
         this->cuda_pointer = ptr.cuda_pointer;
         ptr.pointer = nullptr;
         ptr.pd = nullptr;
         ptr.cuda_pointer = nullptr;
         return *this;
      }

      bool synchronize()
      {
         if( ! this->pd )
            return true;
#ifdef HAVE_CUDA
         if( this->modified() )
         {
            TNL_ASSERT( this->pointer, );
            TNL_ASSERT( this->cuda_pointer, );
            cudaMemcpy( (void*) this->cuda_pointer, (void*) this->pointer, sizeof( ObjectType ), cudaMemcpyHostToDevice );
            if( ! TNL_CHECK_CUDA_DEVICE ) {
               return false;
            }
            this->set_last_sync_state();
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

      struct PointerData
      {
         char data_image[ sizeof(Object) ];
         int counter = 1;
         bool maybe_modified = false;
      };

      bool allocate( ObjectType& obj )
      {
         this->pointer = &obj;
         this->pd = new PointerData();
         // pass to device
         this->cuda_pointer = Devices::Cuda::passToDevice( *this->pointer );
         // set last-sync state
         this->set_last_sync_state();
         Devices::Cuda::insertSmartPointer( this );
         return true;
      }

      void set_last_sync_state()
      {
         TNL_ASSERT( this->pointer, );
         TNL_ASSERT( this->pd, );
         std::memcpy( (void*) &this->pd->data_image, (void*) this->pointer, sizeof( Object ) );
         this->pd->maybe_modified = false;
      }

      bool modified()
      {
         TNL_ASSERT( this->pointer, );
         TNL_ASSERT( this->pd, );
         // optimization: skip bitwise comparison if we're sure that the data is the same
         if( ! this->pd->maybe_modified )
            return false;
         return std::memcmp( (void*) &this->pd->data_image, (void*) this->pointer, sizeof( Object ) ) != 0;
      }

      void free()
      {
         if( this->pd )
         {
            if( ! --this->pd->counter )
            {
               delete this->pd;
               this->pd = nullptr;
               if( this->cuda_pointer )
                  Devices::Cuda::freeFromDevice( this->cuda_pointer );
            }
         }
      }

      Object* pointer;

      PointerData* pd;

      // cuda_pointer can't be part of PointerData structure, since we would be
      // unable to dereference this-pd on the device
      Object* cuda_pointer;
};


#ifndef NDEBUG
namespace Assert {

template< typename Object, typename Device >
struct Formatter< DevicePointer< Object, Device > >
{
   static std::string
   printToString( const DevicePointer< Object, Device >& value )
   {
      ::std::stringstream ss;
      ss << "(DevicePointer< " << Object::getType() << ", " << Device::getDeviceType()
         << " > object at " << &value << ")";
      return ss.str();
   }
};

} // namespace Assert
#endif

} // namespace TNL
