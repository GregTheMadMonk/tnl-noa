/***************************************************************************
                          MeshSubtopology.h  -  description
                             -------------------
    begin                : Aug 29, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
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
namespace Meshes {

template< typename Topology,
          int dimension >
class MeshSubtopology;

template< typename Topology,
          typename Subtopology,
          int subtopologyIndex,
          int vertexIndex >
struct MeshSubtopologyVertex;

} // namespace Meshes
} // namespace TNL

