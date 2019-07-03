/***************************************************************************
                          Host.h  -  description
                             -------------------
    begin                : Apr 8, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <memory>

namespace TNL {

/**
 * \brief Namespace for TNL allocators.
 *
 * All TNL allocators must satisfy the requirements imposed by the
 * [Allocator concept](https://en.cppreference.com/w/cpp/named_req/Allocator)
 * from STL.
 */
namespace Allocators {

/**
 * \brief Allocator for the host memory space -- alias for \ref std::allocator.
 */
template< class T >
using Host = std::allocator< T >;

} // namespace Allocators
} // namespace TNL
