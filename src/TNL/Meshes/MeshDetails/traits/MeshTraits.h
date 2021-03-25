/***************************************************************************
                          MeshTraits.h  -  description
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

#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/Array.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Meshes/DimensionTag.h>
#include <TNL/Meshes/Topologies/Vertex.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig, typename Device, typename EntityTopology > class MeshEntity;
template< typename MeshConfig, typename EntityTopology > class EntitySeed;
template< typename MeshConfig, typename Device, int Dimension > class MeshEntityTraits;
template< typename MeshConfig, typename Device, typename MeshEntity, int Subdimension > class MeshSubentityTraits;
template< typename MeshConfig, typename Device, typename MeshEntity, int Superdimension > class MeshSuperentityTraits;

// helper templates (must be public because nvcc sucks, and outside of MeshTraits to avoid duplicate code generation)
template< typename Device, typename Index, typename IndexAlocator >
using EllpackSegments = Algorithms::Segments::Ellpack< Device, Index, IndexAlocator >;
template< typename Device, typename Index, typename IndexAlocator >
using SlicedEllpackSegments = Algorithms::Segments::SlicedEllpack< Device, Index, IndexAlocator >;

template< typename MeshConfig,
          typename Device = Devices::Host >
class MeshTraits
{
public:
   static constexpr int meshDimension  = MeshConfig::CellTopology::dimension;
   static constexpr int spaceDimension = MeshConfig::spaceDimension;

   using DeviceType          = Device;
   using GlobalIndexType     = typename MeshConfig::GlobalIndexType;
   using LocalIndexType      = typename MeshConfig::LocalIndexType;

   using CellTopology        = typename MeshConfig::CellTopology;
   using CellType            = MeshEntity< MeshConfig, Device, CellTopology >;
   using VertexType          = MeshEntity< MeshConfig, Device, Topologies::Vertex >;
   using PointType           = Containers::StaticVector< spaceDimension, typename MeshConfig::RealType >;
   using CellSeedType        = EntitySeed< MeshConfig, CellTopology >;
   using EntityTagType       = std::uint8_t;

   using NeighborCountsArray = Containers::Vector< LocalIndexType, DeviceType, GlobalIndexType >;
   using PointArrayType      = Containers::Array< PointType, DeviceType, GlobalIndexType >;
   using CellSeedArrayType   = Containers::Array< CellSeedType, DeviceType, GlobalIndexType >;
   using EntityTagsArrayType = Containers::Array< EntityTagType, DeviceType, GlobalIndexType >;

   template< int Dimension >
   using EntityTraits = MeshEntityTraits< MeshConfig, DeviceType, Dimension >;

   template< typename EntityTopology, int Subdimension >
   using SubentityTraits = MeshSubentityTraits< MeshConfig, DeviceType, EntityTopology, Subdimension >;

   template< typename EntityTopology, int Superdimension >
   using SuperentityTraits = MeshSuperentityTraits< MeshConfig, DeviceType, EntityTopology, Superdimension >;

   using DimensionTag = Meshes::DimensionTag< meshDimension >;

   // container for storing the subentity indices
   template< int Dimension, int Subdimension >
   using SubentityMatrixType = typename SubentityTraits< typename EntityTraits< Dimension >::EntityTopology, Subdimension >::SubentityMatrixType;

   // container for storing the superentity indices
   using SuperentityMatrixType = Matrices::SparseMatrix< bool, Device, GlobalIndexType, Matrices::GeneralMatrix, SlicedEllpackSegments >;

   // container for storing the dual graph adjacency matrix
   using DualGraph = Matrices::SparseMatrix< bool, Device, GlobalIndexType, Matrices::GeneralMatrix, SlicedEllpackSegments >;
};

} // namespace Meshes
} // namespace TNL
