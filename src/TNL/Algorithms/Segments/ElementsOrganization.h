// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Devices/Host.h>

namespace TNL {
   namespace Algorithms {
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
} // namespace Algorithms

inline std::string getSerializationType( Algorithms::Segments::ElementsOrganization Organization )
{
   if( Organization == Algorithms::Segments::RowMajorOrder )
      return "RowMajorOrder";
   return "ColumnMajorOrder";
}

} // namespace TNL
