/***************************************************************************
                          SharedPointerCuda.h  -  description
                             -------------------
    begin                : Aug 22, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include "SharedPointer.h"

#include <TNL/Devices/Cuda.h>
#include <TNL/Pointers/SmartPointer.h>

#include <cstring>   // std::memcpy, std::memcmp
#include <cstddef>   // std::nullptr_t
#include <algorithm> // swap

namespace TNL {
namespace Pointers {

//#define HAVE_CUDA_UNIFIED_MEMORY

#ifdef HAVE_CUDA_UNIFIED_MEMORY
template< typename Object >
class SharedPointer< Object, Devices::Cuda > : public SmartPointer
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
      friend class SharedPointer;

   public:

      using ObjectType = Object;
      using DeviceType = Devices::Cuda; 

      SharedPointer( std::nullptr_t )
      : pd( nullptr )
      {}

      template< typename... Args >
      explicit  SharedPointer( Args... args )
      : pd( nullptr )
      {
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Creating shared pointer to " << getType< ObjectType >() << std::endl;
#endif
         this->allocate( args... );
      }

      // this is needed only to avoid the default compiler-generated constructor
      SharedPointer( const SharedPointer& pointer )
      : pd( (PointerData*) pointer.pd )
      {
         this->pd->counter += 1;
      }

      // conditional constructor for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      SharedPointer( const SharedPointer<  Object_, DeviceType >& pointer )
      : pd( (PointerData*) pointer.pd )
      {
         this->pd->counter += 1;
      }

      // this is needed only to avoid the default compiler-generated constructor
      SharedPointer( SharedPointer&& pointer )
      : pd( (PointerData*) pointer.pd )
      {
         pointer.pd = nullptr;
      }

      // conditional constructor for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      SharedPointer( SharedPointer<  Object_, DeviceType >&& pointer )
      : pd( (PointerData*) pointer.pd )
      {
         pointer.pd = nullptr;
      }

      template< typename... Args >
      bool recreate( Args... args )
      {
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Recreating shared pointer to " << getType< ObjectType >() << std::endl;
#endif
         if( ! this->counter )
            return this->allocate( args... );

         if( *this->pd->counter == 1 )
         {
            /****
             * The object is not shared -> recreate it in-place, without reallocation
             */
            this->pd->data.~ObjectType();
            new ( this->pd->data ) ObjectType( args... );
            return true;
         }

         // free will just decrement the counter
         this->free();

         return this->allocate( args... );
      }

      const Object* operator->() const
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         return &this->pd->data;
      }

      Object* operator->()
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         return &this->pd->data;
      }

      const Object& operator *() const
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         return this->pd->data;
      }

      Object& operator *()
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         return this->pd->data;
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
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         return this->pd->data;
      }

      template< typename Device = Devices::Host >
      __cuda_callable__
      Object& modifyData()
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         return this->pd->data;
      }

      // this is needed only to avoid the default compiler-generated operator
      const SharedPointer& operator=( const SharedPointer& ptr )
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         if( this->pd != nullptr ) 
            this->pd->counter += 1;
         return *this;
      }

      // conditional operator for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      const SharedPointer& operator=( const SharedPointer<  Object_, DeviceType >& ptr )
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         if( this->pd != nullptr )
            this->pd->counter += 1;
         return *this;
      }

      // this is needed only to avoid the default compiler-generated operator
      const SharedPointer& operator=( SharedPointer&& ptr )
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         ptr.pd = nullptr;
         return *this;
      }

      // conditional operator for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      const SharedPointer& operator=( SharedPointer<  Object_, DeviceType >&& ptr )
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         ptr.pd = nullptr;
         return *this;
      }

      bool synchronize()
      {
         return true;
      }

      void clear()
      {
         this->free();
      }

      void swap( SharedPointer& ptr2 )
      {
         std::swap( this->pd, ptr2.pd );
      }

      ~SharedPointer()
      {
         this->free();
      }


   protected:

      struct PointerData
      {
         Object data;
         int counter;

         template< typename... Args >
         explicit PointerData( Args... args )
         : data( args... ),
           counter( 1 )
         {}
      };

      template< typename... Args >
      bool allocate( Args... args )
      {
#ifdef HAVE_CUDA
         if( cudaMallocManaged( ( void** ) &this->pd, sizeof( PointerData ) != cudaSuccess ) )
            return false;
         new ( this->pd ) PointerData( args... );
         return true;
#else
         return false;
#endif
      }

      void free()
      {
         if( this->pd )
         {
            if( ! --this->pd->counter )
            {
#ifdef HAVE_CUDA
               cudaFree( this->pd );
#endif
               this->pd = nullptr;
            }
         }
      }

      PointerData* pd;
};

#else // HAVE_CUDA_UNIFIED_MEMORY

template< typename Object >
class SharedPointer< Object, Devices::Cuda > : public SmartPointer
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
      friend class SharedPointer;

   public:

      using ObjectType = Object;
      using DeviceType = Devices::Cuda; 

      SharedPointer( std::nullptr_t )
      : pd( nullptr ),
        cuda_pointer( nullptr )
      {}

      template< typename... Args >
      explicit  SharedPointer( Args... args )
      : pd( nullptr ),
        cuda_pointer( nullptr )
      {
         this->allocate( args... );
      }

      // this is needed only to avoid the default compiler-generated constructor
      SharedPointer( const SharedPointer& pointer )
      : pd( (PointerData*) pointer.pd ),
        cuda_pointer( pointer.cuda_pointer )
      {
         this->pd->counter += 1;
      }

      // conditional constructor for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      SharedPointer( const SharedPointer<  Object_, DeviceType >& pointer )
      : pd( (PointerData*) pointer.pd ),
        cuda_pointer( pointer.cuda_pointer )
      {
         this->pd->counter += 1;
      }

      // this is needed only to avoid the default compiler-generated constructor
      SharedPointer( SharedPointer&& pointer )
      : pd( (PointerData*) pointer.pd ),
        cuda_pointer( pointer.cuda_pointer )
      {
         pointer.pd = nullptr;
         pointer.cuda_pointer = nullptr;
      }

      // conditional constructor for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      SharedPointer( SharedPointer<  Object_, DeviceType >&& pointer )
      : pd( (PointerData*) pointer.pd ),
        cuda_pointer( pointer.cuda_pointer )
      {
         pointer.pd = nullptr;
         pointer.cuda_pointer = nullptr;
      }

      template< typename... Args >
      bool recreate( Args... args )
      {
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Recreating shared pointer to " << getType< ObjectType >() << std::endl;
#endif
         if( ! this->pd )
            return this->allocate( args... );

         if( this->pd->counter == 1 )
         {
            /****
             * The object is not shared -> recreate it in-place, without reallocation
             */
            this->pd->data.~Object();
            new ( &this->pd->data ) Object( args... );
#ifdef HAVE_CUDA
            cudaMemcpy( (void*) this->cuda_pointer, (void*) &this->pd->data, sizeof( Object ), cudaMemcpyHostToDevice );
#endif
            this->set_last_sync_state();
            return true;
         }

         // free will just decrement the counter
         this->free();

         return this->allocate( args... );
      }

      const Object* operator->() const
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         return &this->pd->data;
      }

      Object* operator->()
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         this->pd->maybe_modified = true;
         return &this->pd->data;
      }

      const Object& operator *() const
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         return this->pd->data;
      }

      Object& operator *()
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         this->pd->maybe_modified = true;
         return this->pd->data;
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
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         TNL_ASSERT_TRUE( this->cuda_pointer, "Attempt to dereference a null pointer" );
         if( std::is_same< Device, Devices::Host >::value )
            return this->pd->data;
         if( std::is_same< Device, Devices::Cuda >::value )
            return *( this->cuda_pointer );
      }

      template< typename Device = Devices::Host >
      __cuda_callable__
      Object& modifyData()
      {
         static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value, "Only Devices::Host or Devices::Cuda devices are accepted here." );
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         TNL_ASSERT_TRUE( this->cuda_pointer, "Attempt to dereference a null pointer" );
         if( std::is_same< Device, Devices::Host >::value )
         {
            this->pd->maybe_modified = true;
            return this->pd->data;
         }
         if( std::is_same< Device, Devices::Cuda >::value )
            return *( this->cuda_pointer );
      }

      // this is needed only to avoid the default compiler-generated operator
      const SharedPointer& operator=( const SharedPointer& ptr )
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         this->cuda_pointer = ptr.cuda_pointer;
         if( this->pd != nullptr )
            this->pd->counter += 1;
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Copy-assigned shared pointer: counter = " << this->pd->counter << ", type: " << getType< ObjectType >() << std::endl;
#endif
         return *this;
      }

      // conditional operator for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      const SharedPointer& operator=( const SharedPointer<  Object_, DeviceType >& ptr )
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         this->cuda_pointer = ptr.cuda_pointer;
         if( this->pd != nullptr )
            this->pd->counter += 1;
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Copy-assigned shared pointer: counter = " << this->pd->counter << ", type: " << getType< ObjectType >() << std::endl;
#endif
         return *this;
      }

      // this is needed only to avoid the default compiler-generated operator
      const SharedPointer& operator=( SharedPointer&& ptr )
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         this->cuda_pointer = ptr.cuda_pointer;
         ptr.pd = nullptr;
         ptr.cuda_pointer = nullptr;
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Move-assigned shared pointer: counter = " << this->pd->counter << ", type: " << getType< ObjectType >() << std::endl;
#endif
         return *this;
      }

      // conditional operator for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      const SharedPointer& operator=( SharedPointer<  Object_, DeviceType >&& ptr )
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         this->cuda_pointer = ptr.cuda_pointer;
         ptr.pd = nullptr;
         ptr.cuda_pointer = nullptr;
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Move-assigned shared pointer: counter = " << this->pd->counter << ", type: " << getType< ObjectType >() << std::endl;
#endif
         return *this;
      }

      bool synchronize()
      {
         if( ! this->pd )
            return true;
#ifdef HAVE_CUDA
         if( this->modified() )
         {
#ifdef TNL_DEBUG_SHARED_POINTERS
            std::cerr << "Synchronizing shared pointer: counter = " << this->pd->counter << ", type: " << getType< ObjectType >() << std::endl;
            std::cerr << "   ( " << sizeof( Object ) << " bytes, CUDA adress " << this->cuda_pointer << " )" << std::endl;
#endif
            TNL_ASSERT( this->cuda_pointer, );
            cudaMemcpy( (void*) this->cuda_pointer, (void*) &this->pd->data, sizeof( Object ), cudaMemcpyHostToDevice );
            TNL_CHECK_CUDA_DEVICE;
            this->set_last_sync_state();
            return true;
         }
         return true;
#else
         return false;
#endif
      }

      void clear()
      {
         this->free();
      }

      void swap( SharedPointer& ptr2 )
      {
         std::swap( this->pd, ptr2.pd );
         std::swap( this->cuda_pointer, ptr2.cuda_pointer );
      }

      ~SharedPointer()
      {
         this->free();
         Devices::Cuda::removeSmartPointer( this );
      }

   protected:

      struct PointerData
      {
         Object data;
         char data_image[ sizeof(Object) ];
         int counter;
         bool maybe_modified;

         template< typename... Args >
         explicit PointerData( Args... args )
         : data( args... ),
           counter( 1 ),
           maybe_modified( false )
         {}
      };

      template< typename... Args >
      bool allocate( Args... args )
      {
         this->pd = new PointerData( args... );
         // pass to device
         this->cuda_pointer = Devices::Cuda::passToDevice( this->pd->data );
         // set last-sync state
         this->set_last_sync_state();
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Created shared pointer to " << getType< ObjectType >() << " (cuda_pointer = " << this->cuda_pointer << ")" << std::endl;
#endif
         Devices::Cuda::insertSmartPointer( this );
         return true;
      }

      void set_last_sync_state()
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         std::memcpy( (void*) &this->pd->data_image, (void*) &this->pd->data, sizeof( Object ) );
         this->pd->maybe_modified = false;
      }

      bool modified()
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         // optimization: skip bitwise comparison if we're sure that the data is the same
         if( ! this->pd->maybe_modified )
            return false;
         return std::memcmp( (void*) &this->pd->data_image, (void*) &this->pd->data, sizeof( Object ) ) != 0;
      }

      void free()
      {
         if( this->pd )
         {
#ifdef TNL_DEBUG_SHARED_POINTERS
            std::cerr << "Freeing shared pointer: counter = " << this->pd->counter << ", cuda_pointer = " << this->cuda_pointer << ", type: " << getType< ObjectType >() << std::endl;
#endif
            if( ! --this->pd->counter )
            {
               delete this->pd;
               this->pd = nullptr;
               if( this->cuda_pointer )
                  Devices::Cuda::freeFromDevice( this->cuda_pointer );
#ifdef TNL_DEBUG_SHARED_POINTERS
               std::cerr << "...deleted data." << std::endl;
#endif
            }
         }
      }

      PointerData* pd;

      // cuda_pointer can't be part of PointerData structure, since we would be
      // unable to dereference this-pd on the device
      Object* cuda_pointer;
};
#endif // HAVE_CUDA_UNIFIED_MEMORY

} // namespace Pointers
} // namespace TNL
