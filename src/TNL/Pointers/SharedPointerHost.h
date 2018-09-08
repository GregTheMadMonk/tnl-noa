/***************************************************************************
                          SharedPointerHost.h  -  description
                             -------------------
    begin                : Aug 22, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include "SharedPointer.h"

#include <TNL/Devices/Host.h>
#include <TNL/Devices/CudaCallable.h>
#include <TNL/Pointers/SmartPointer.h>

#include <cstddef>  // std::nullptr_t

namespace TNL {
namespace Pointers {

template< typename Object >
class SharedPointer< Object, Devices::Host > : public SmartPointer
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

      typedef Object ObjectType;
      typedef Devices::Host DeviceType;
      typedef SharedPointer<  Object, Devices::Host > ThisType;

      SharedPointer( std::nullptr_t )
      : pd( nullptr )
      {}

      template< typename... Args >
      explicit  SharedPointer( Args... args )
      : pd( nullptr )
      {
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Creating shared pointer to " << demangle(typeid(ObjectType).name()) << std::endl;
#endif
         this->allocate( args... );
      }

      // this is needed only to avoid the default compiler-generated constructor
      SharedPointer( const ThisType& pointer )
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
      SharedPointer( ThisType&& pointer )
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
         std::cerr << "Recreating shared pointer to " << demangle(typeid(ObjectType).name()) << std::endl;
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
         return &this->pd->data;
      }

      Object* operator->()
      {
         return &this->pd->data;
      }

      const Object& operator *() const
      {
         return this->pd->data;
      }

      Object& operator *()
      {
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
         return this->pd->data;
      }

      template< typename Device = Devices::Host >
      __cuda_callable__
      Object& modifyData()
      {
         return this->pd->data;
      }

      // this is needed only to avoid the default compiler-generated operator
      const ThisType& operator=( const ThisType& ptr )
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         this->pd->counter += 1;
         return *this;
      }

      // conditional operator for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      const ThisType& operator=( const SharedPointer<  Object_, DeviceType >& ptr )
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         this->pd->counter += 1;
         return *this;
      }

      // this is needed only to avoid the default compiler-generated operator
      const ThisType& operator=( ThisType&& ptr )
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         ptr.pd = nullptr;
         return *this;
      }

      // conditional operator for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      const ThisType& operator=( SharedPointer<  Object_, DeviceType >&& ptr )
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
         this->pd = new PointerData( args... );
         return this->pd;
      }

      void free()
      {
         if( this->pd )
         {
            if( ! --this->pd->counter )
            {
               delete this->pd;
               this->pd = nullptr;
            }
         }

      }

      PointerData* pd;
};

} // namespace Pointers
} // namespace TNL
