/***************************************************************************
                          DistributedMesh.h  -  description
                             -------------------
    begin                : April 11, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

namespace TNL {
namespace Meshes {
namespace DistributedMeshes {

template< typename Mesh, int Dimension >
class GlobalIndexStorage
{
public:
   using GlobalIndexArray = typename Mesh::GlobalIndexArray;

   const GlobalIndexArray&
   getGlobalIndices() const
   {
      return globalIndices;
   }

   GlobalIndexArray&
   getGlobalIndices()
   {
      return globalIndices;
   }

protected:
   GlobalIndexArray globalIndices;
};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL
