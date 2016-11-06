/***************************************************************************
                          MeshVertexTopology.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <TNL/String.h>

namespace TNL {
namespace Meshes {

struct MeshVertexTopology
{
   static constexpr int dimensions = 0;

   static String getType()
   {
      return "MeshVertexTopology";
   }
};

} // namespace Meshes
} // namespace TNL
