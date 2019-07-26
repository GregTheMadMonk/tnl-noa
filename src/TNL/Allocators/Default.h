/***************************************************************************
                          Default.h  -  description
                             -------------------
    begin                : Jul 2, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Allocators/Host.h>
#include <TNL/Allocators/Cuda.h>
#include <TNL/Allocators/MIC.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/MIC.h>

namespace TNL {
namespace Allocators {

/**
 * \brief A trait-like class used for the selection of a default allocators for
 * given device.
 */
template< typename Device >
struct Default;

//! Sets \ref Allocators::Host as the default allocator for \ref Devices::Host.
template<>
struct Default< Devices::Host >
{
   template< typename T >
   using Allocator = Allocators::Host< T >;
};

//! Sets \ref Allocators::Cuda as the default allocator for \ref Devices::Cuda.
template<>
struct Default< Devices::Cuda >
{
   template< typename T >
   using Allocator = Allocators::Cuda< T >;
};

//! Sets \ref Allocators::MIC as the default allocator for \ref Devices::MIC.
template<>
struct Default< Devices::MIC >
{
   template< typename T >
   using Allocator = Allocators::MIC< T >;
};

} // namespace Allocators
} // namespace TNL
