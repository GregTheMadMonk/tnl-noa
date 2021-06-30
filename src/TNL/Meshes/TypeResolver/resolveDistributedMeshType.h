/***************************************************************************
                          resolveDistributedMesh.h  -  description
                             -------------------
    begin                : Apr 9, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>

namespace TNL {
namespace Meshes {

template< typename ConfigTag,
          typename Device,
          typename Functor >
bool
resolveDistributedMeshType( Functor&& functor,
                            const std::string& fileName,
                            const std::string& fileFormat = "auto" );

template< typename ConfigTag,
          typename Device,
          typename Functor >
bool
resolveAndLoadDistributedMesh( Functor&& functor,
                               const std::string& fileName,
                               const std::string& fileFormat = "auto" );

template< typename Mesh >
bool
loadDistributedMesh( DistributedMeshes::DistributedMesh< Mesh >& distributedMesh,
                     const std::string& fileName,
                     const std::string& fileFormat = "auto" );

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/TypeResolver/resolveDistributedMeshType.hpp>
