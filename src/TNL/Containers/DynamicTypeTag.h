/***************************************************************************
                          DynamicTypeTag.h  -  description
                             -------------------
    begin                : Mar 13, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Containers {   

template< typename Element >
struct DynamicTypeTag
{
   enum { value = false };
};

} // namespace Containers
} // namespace TNL
