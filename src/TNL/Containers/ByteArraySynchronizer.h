/***************************************************************************
                          ByteArraySynchronizer.h  -  description
                             -------------------
    begin                : November 17, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/Containers/ArrayView.h>

namespace TNL {
namespace Containers {

template< typename Device, typename Index >
class ByteArraySynchronizer
{
public:
   using ByteArrayView = ArrayView< std::uint8_t, Device, Index >;

   virtual void synchronizeByteArray( ByteArrayView& array, int bytesPerValue ) = 0;

   virtual ~ByteArraySynchronizer() = default;
};

} // namespace Containers
} // namespace TNL
