/***************************************************************************
                          Segments.h -  description
                             -------------------
    begin                : Nov 29, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Containers {

template< typename Value,
          typename Organization >
class Segments
{
   public:

      using ValueType = Value;
      using OrganizationType = Organization;
      using IndexType = typename Organization::IndexType;

};

}  // namespace Conatiners
} // namespace TNL