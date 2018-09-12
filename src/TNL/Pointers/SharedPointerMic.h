/***************************************************************************
                          SharedPointerMic.h  -  description
                             -------------------
    begin                : Aug 22, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include "SharedPointer.h"

#include <TNL/Devices/MIC.h>
#include <TNL/Pointers/SmartPointer.h>

#include <cstring>   // std::memcpy, std::memcmp
#include <cstddef>   // std::nullptr_t
#include <algorithm> // swap

namespace TNL {
namespace Pointers {

#ifdef HAVE_MIC
template< typename Object>
class SharedPointer< Object, Devices::MIC > : public SmartPointer
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
      template< typename Object_, typename Device_>
      friend class SharedPointer;

   public:

      using ObjectType = Object;
      using DeviceType = Devices::MIC; 
      using ThisType = SharedPointer<  Object, Devices::MIC >;

      SharedPointer( std::nullptr_t )
      : pd( nullptr ),
        mic_pointer( nullptr )
      {}

      template< typename... Args >
      explicit  SharedPointer( Args... args )
      : pd( nullptr ),
        mic_pointer( nullptr )
      {
            this->allocate( args... );
      }

      // this is needed only to avoid the default compiler-generated constructor
      SharedPointer( const ThisType& pointer )
      : pd( (PointerData*) pointer.pd ),
        mic_pointer( pointer.mic_pointer )
      {
         this->pd->counter += 1;
      }

      // conditional constructor for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      SharedPointer( const SharedPointer< Object_, DeviceType >& pointer )
      : pd( (PointerData*) pointer.pd ),
        mic_pointer( pointer.mic_pointer )
      {
         this->pd->counter += 1;
      }

      // this is needed only to avoid the default compiler-generated constructor
      SharedPointer( ThisType&& pointer )
      : pd( (PointerData*) pointer.pd ),
        mic_pointer( pointer.mic_pointer )
      {
         pointer.pd = nullptr;
         pointer.mic_pointer = nullptr;
      }

      // conditional constructor for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      SharedPointer( SharedPointer< Object_, DeviceType >&& pointer )
      : pd( (PointerData*) pointer.pd ),
        mic_pointer( pointer.mic_pointer )
      {
         pointer.pd = nullptr;
         pointer.mic_pointer = nullptr;
      }

      template< typename... Args >
      bool recreate( Args... args )
      {
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Recreating shared pointer to " << demangle(typeid(ObjectType).name()) << std::endl;
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
            Devices::MIC::CopyToMIC(this->mic_pointer,(void*) &this->pd->data,sizeof(Object));
            this->set_last_sync_state();
            return true;
         }

         // free will just decrement the counter
         this->free();

         return this->allocate( args... );
      }

      const Object* operator->() const
      {
         TNL_ASSERT_TRUE( this->pd != nullptr, "Attempt of dereferencing of null pointer" );
         return &this->pd->data;
      }

      Object* operator->()
      {
         TNL_ASSERT_TRUE( this->pd != nullptr, "Attempt of dereferencing of null pointer" );
         this->pd->maybe_modified = true;
         return &this->pd->data;
      }

      const Object& operator *() const
      {
         TNL_ASSERT_TRUE( this->pd != nullptr, "Attempt of dereferencing of null pointer" );
         return this->pd->data;
      }

      Object& operator *()
      {
         TNL_ASSERT_TRUE( this->pd != nullptr, "Attempt of dereferencing of null pointer" );
         this->pd->maybe_modified = true;
         return this->pd->data;
      }

      operator bool()
      {
         return this->pd;
      }

      template< typename Device = Devices::Host >
      __cuda_callable__
      const Object& getData() const
      {
         static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::MIC >::value, "Only Devices::Host or Devices::MIC devices are accepted here." );
         TNL_ASSERT( this->pd, );
         TNL_ASSERT( this->mic_pointer, );
         if( std::is_same< Device, Devices::Host >::value )
            return this->pd->data;
         if( std::is_same< Device, Devices::MIC >::value )
            return *( this->mic_pointer );

      }

      template< typename Device = Devices::Host >
      __cuda_callable__
      Object& modifyData()
      {
         static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::MIC >::value, "Only Devices::Host or Devices::MIC devices are accepted here." );
         TNL_ASSERT( this->pd, );
         TNL_ASSERT( this->mic_pointer, );
         if( std::is_same< Device, Devices::Host >::value )
         {
            this->pd->maybe_modified = true;
            return this->pd->data;
         }
         if( std::is_same< Device, Devices::MIC >::value )
            return *( this->mic_pointer );

      }

      // this is needed only to avoid the default compiler-generated operator
      const ThisType& operator=( const ThisType& ptr )
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         this->mic_pointer = ptr.mic_pointer;
         if( this->pd != nullptr )
            this->pd->counter += 1;
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Copy-assigned shared pointer: counter = " << this->pd->counter << ", type: " << demangle(typeid(ObjectType).name()) << std::endl;
#endif
         return *this;
      }

      // conditional operator for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      const ThisType& operator=( const SharedPointer< Object_, DeviceType >& ptr )
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         this->mic_pointer = ptr.mic_pointer;
         if( this->pd != nullptr )
            this->pd->counter += 1;
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Copy-assigned shared pointer: counter = " << this->pd->counter << ", type: " << demangle(typeid(ObjectType).name()) << std::endl;
#endif
         return *this;
      }

      // this is needed only to avoid the default compiler-generated operator
      const ThisType& operator=( ThisType&& ptr )
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         this->mic_pointer = ptr.mic_pointer;
         ptr.pd = nullptr;
         ptr.mic_pointer = nullptr;
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Move-assigned shared pointer: counter = " << this->pd->counter << ", type: " << demangle(typeid(ObjectType).name()) << std::endl;
#endif
         return *this;
      }

      // conditional operator for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      const ThisType& operator=( SharedPointer< Object_, DeviceType >&& ptr )
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         this->mic_pointer = ptr.mic_pointer;
         ptr.pd = nullptr;
         ptr.mic_pointer = nullptr;
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Move-assigned shared pointer: counter = " << this->pd->counter << ", type: " << demangle(typeid(ObjectType).name()) << std::endl;
#endif
         return *this;
      }

      bool synchronize()
      {
         if( ! this->pd )
            return true;

         if( this->modified() )
         {
#ifdef TNL_DEBUG_SHARED_POINTERS
            std::cerr << "Synchronizing shared pointer: counter = " << this->pd->counter << ", type: " << demangle(typeid(Object).name()) << std::endl;
            std::cerr << "   ( " << sizeof( Object ) << " bytes, MIC adress " << this->mic_pointer << " )" << std::endl;
#endif
            TNL_ASSERT( this->mic_pointer, );

            Devices::MIC::CopyToMIC((void*)this->mic_pointer,(void*) &this->pd->data,sizeof(Object));
            this->set_last_sync_state();
            return true;
         }
         return false; //??
      }

      void clear()
      {
         this->free();
      }

      void swap( ThisType& ptr2 )
      {
         std::swap( this->pd, ptr2.pd );
         std::swap( this->mic_pointer, ptr2.mic_pointer );
      }

      ~SharedPointer()
      {
         this->free();
         Devices::MIC::removeSmartPointer( this );
      }

   protected:

      struct PointerData
      {
         Object data;
         uint8_t data_image[ sizeof(Object) ];
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
         if( ! this->pd )
            return false;

         mic_pointer=(Object*)Devices::MIC::AllocMIC(sizeof(Object));
         Devices::MIC::CopyToMIC((void*)this->mic_pointer,(void*) &this->pd->data,sizeof(Object));

         if( ! this->mic_pointer )
            return false;
         // set last-sync state
         this->set_last_sync_state();
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Created shared pointer to " << demangle(typeid(ObjectType).name()) << " (mic_pointer = " << this->mic_pointer << ")" << std::endl;
#endif
         Devices::MIC::insertSmartPointer( this );
         return true;
      }

      void set_last_sync_state()
      {
         TNL_ASSERT( this->pd, );
         std::memcpy( (void*) &this->pd->data_image, (void*) &this->pd->data, sizeof( Object ) );
         this->pd->maybe_modified = false;
      }

      bool modified()
      {
         TNL_ASSERT( this->pd, );
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
            std::cerr << "Freeing shared pointer: counter = " << this->pd->counter << ", mic_pointer = " << this->mic_pointer << ", type: " << demangle(typeid(ObjectType).name()) << std::endl;
#endif
            if( ! --this->pd->counter )
            {
               delete this->pd;
               this->pd = nullptr;
               if( this->mic_pointer )
               {
                   Devices::MIC::FreeMIC((void*)mic_pointer);
                   mic_pointer=nullptr;
               }
#ifdef TNL_DEBUG_SHARED_POINTERS
               std::cerr << "...deleted data." << std::endl;
#endif
            }
         }
      }

      PointerData* pd;

      // cuda_pointer can't be part of PointerData structure, since we would be
      // unable to dereference this-pd on the device -- Nevím zda to platí pro MIC, asi jo
      Object* mic_pointer;
};
#endif

} // namespace Pointers
} // namespace TNL
