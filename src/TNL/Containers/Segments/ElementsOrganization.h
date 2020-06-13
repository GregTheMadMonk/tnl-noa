/***************************************************************************
                          ElementsOrganization.h -  description
                             -------------------
    begin                : May 9, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Devices/Host.h>

namespace TNL {
namespace Containers {
namespace Segments {

enum ElementsOrganization { ColumnMajorOrder = 0, RowMajorOrder };

template< typename Device >
struct DefaultElementsOrganization
{
   static constexpr ElementsOrganization getOrganization() {
      if( std::is_same< Device, Devices::Host >::value )
         return RowMajorOrder;
      else
         return ColumnMajorOrder;
   };
};

} // namespace Segments
} // namespace Containers

inline String getSerializationType( Containers::Segments::ElementsOrganization Organization )
{
   if( Organization == Containers::Segments::RowMajorOrder )
      return String( "RowMajorOrder" );
   else
      return String( "ColumnMajorOrder" );
}

} // namespace TNL
