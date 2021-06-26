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

template< typename MeshConfig,
          typename Device >
bool
loadDistributedMesh( Mesh< MeshConfig, Device >& mesh,
                     DistributedMeshes::DistributedMesh< Mesh< MeshConfig, Device > >& distributedMesh,
                     const std::string& fileName,
                     const std::string& fileFormat = "auto" );

// overloads for grids
template< int Dimension,
          typename Real,
          typename Device,
          typename Index >
bool
loadDistributedMesh( Grid< Dimension, Real, Device, Index >& mesh,
                     DistributedMeshes::DistributedMesh< Grid< Dimension, Real, Device, Index > > &distributedMesh,
                     const std::string& fileName,
                     const std::string& fileFormat = "auto" );

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/TypeResolver/resolveDistributedMeshType.hpp>
