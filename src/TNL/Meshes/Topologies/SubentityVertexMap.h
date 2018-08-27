/***************************************************************************
                          SubentityVertexMap.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

namespace TNL {
namespace Meshes{
namespace Topologies {

template< typename EntityTopology,
          int SubentityDimension >
struct Subtopology
{
};

template< typename EntityTopology,
          typename SubentityTopology,
          int SubentityIndex,
          int SubentityVertexIndex >
struct SubentityVertexMap
{
};

} // namespace Topologies
} // namespace Meshes
} // namespace TNL
