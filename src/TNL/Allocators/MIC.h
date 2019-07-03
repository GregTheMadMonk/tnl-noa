/***************************************************************************
                          MIC.h  -  description
                             -------------------
    begin                : Jul 2, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Devices/MIC.h>
#include <TNL/Exceptions/MICSupportMissing.h>

namespace TNL {
namespace Allocators {

/**
 * \brief Allocator for the MIC device memory space.
 */
template< class T >
struct MIC
{
   using value_type = T;
   using size_type = std::size_t;
   using difference_type = std::ptrdiff_t;

   MIC() = default;
   MIC( const MIC& ) = default;
   MIC( MIC&& ) = default;

   MIC& operator=( const MIC& ) = default;
   MIC& operator=( MIC&& ) = default;

   template< class U >
   MIC( const MIC< U >& )
   {}

   template< class U >
   MIC( MIC< U >&& )
   {}

   template< class U >
   MIC& operator=( const MIC< U >& )
   {
      return *this;
   }

   template< class U >
   MIC& operator=( MIC< U >&& )
   {
      return *this;
   }

   value_type* allocate( size_type size )
   {
#ifdef HAVE_MIC
      Devices::MICHider<void> hide_ptr;
      #pragma offload target(mic) out(hide_ptr) in(size)
      {
         hide_ptr.pointer = malloc(size * sizeof(value_type));
      }
      return hide_ptr.pointer;
#else
      throw Exceptions::MICSupportMissing();
#endif
   }

   void deallocate(value_type* ptr, size_type)
   {
#ifdef HAVE_MIC
      Devices::MICHider<void> hide_ptr;
      hide_ptr.pointer=ptr;
      #pragma offload target(mic) in(hide_ptr)
      {
         free(hide_ptr.pointer);
      }
#else
      throw Exceptions::MICSupportMissing();
#endif
   }
};

template<class T1, class T2>
bool operator==(const MIC<T1>&, const MIC<T2>&)
{
   return true;
}

template<class T1, class T2>
bool operator!=(const MIC<T1>& lhs, const MIC<T2>& rhs)
{
   return !(lhs == rhs);
}

} // namespace Allocators
} // namespace TNL
