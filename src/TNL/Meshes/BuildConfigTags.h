/***************************************************************************
                          BuildConfigTags.h  -  description
                             -------------------
    begin                : Nov 22, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/DefaultConfig.h>
#include <TNL/Meshes/Topologies/Edge.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/Topologies/Quadrilateral.h>
#include <TNL/Meshes/Topologies/Tetrahedron.h>
#include <TNL/Meshes/Topologies/Hexahedron.h>
#include <TNL/Meshes/Topologies/Simplex.h>

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

/****
 * Configuration for structured grids
 */

// 1, 2, and 3 dimensions are enabled by default
template< typename ConfigTag, int Dimension > struct GridDimensionTag { enum { enabled = ( Dimension > 0 && Dimension <= 3 ) }; };

// Grids are enabled for all real types by default.
template< typename ConfigTag, typename Real > struct GridRealTag { enum { enabled = true }; };

// Grids are enabled on all available devices by default.
template< typename ConfigTag, typename Device > struct GridDeviceTag { enum { enabled = true }; };
#ifndef HAVE_CUDA
template< typename ConfigTag > struct GridDeviceTag< ConfigTag, Devices::Cuda > { enum { enabled = false }; };
#endif

// Grids are enabled for all index types by default.
template< typename ConfigTag, typename Index > struct GridIndexTag { enum { enabled = true }; };

// The Grid is enabled for allowed dimensions and Real, Device and Index types.
// 
// By specializing this tag you can enable or disable custom combinations of
// the grid template parameters. The default configuration is identical to the
// individual per-type tags.
template< typename ConfigTag, typename MeshType > struct GridTag { enum { enabled = false }; };

template< typename ConfigTag, int Dimension, typename Real, typename Device, typename Index >
struct GridTag< ConfigTag, Grid< Dimension, Real, Device, Index > >
{
   enum { enabled = GridDimensionTag< ConfigTag, Dimension >::enabled  &&
                    GridRealTag< ConfigTag, Real >::enabled &&
                    GridDeviceTag< ConfigTag, Device >::enabled &&
                    GridIndexTag< ConfigTag, Index >::enabled
   };
};


/****
 * Configuration for unstructured meshes
 */

// Meshes are enabled on all available devices by default.
template< typename ConfigTag, typename Device > struct MeshDeviceTag { enum { enabled = false }; };
template< typename ConfigTag > struct MeshDeviceTag< ConfigTag, Devices::Host > { enum { enabled = true }; };
#ifdef HAVE_CUDA
template< typename ConfigTag > struct MeshDeviceTag< ConfigTag, Devices::Cuda > { enum { enabled = true }; };
#endif

// All available cell topologies are disabled by default.
template< typename ConfigTag, typename CellTopology > struct MeshCellTopologyTag { enum { enabled = false }; };
//template< typename ConfigTag > struct MeshCellTopologyTag< ConfigTag, Topologies::Edge > { enum { enabled = true }; };
//template< typename ConfigTag > struct MeshCellTopologyTag< ConfigTag, Topologies::Triangle > { enum { enabled = true }; };
//template< typename ConfigTag > struct MeshCellTopologyTag< ConfigTag, Topologies::Quadrilateral > { enum { enabled = true }; };
//template< typename ConfigTag > struct MeshCellTopologyTag< ConfigTag, Topologies::Tetrahedron > { enum { enabled = true }; };
//template< typename ConfigTag > struct MeshCellTopologyTag< ConfigTag, Topologies::Hexahedron > { enum { enabled = true }; };
// TODO: Simplex has not been tested yet
//template< typename ConfigTag > struct MeshCellTopologyTag< ConfigTag, Topologies::Simplex > { enum { enabled = true }; };

// All sensible world dimensions are enabled by default.
template< typename ConfigTag, typename CellTopology, int WorldDimension > struct MeshWorldDimensionTag { enum { enabled = ( WorldDimension >= CellTopology::dimension && WorldDimension <= 3 ) }; };

// Meshes are enabled for all real types by default.
template< typename ConfigTag, typename Real > struct MeshRealTag { enum { enabled = true }; };

// Meshes are enabled for all global index types by default.
template< typename ConfigTag, typename GlobalIndex > struct MeshGlobalIndexTag { enum { enabled = true }; };

// Meshes are enabled for all local index types by default.
template< typename ConfigTag, typename LocalIndex > struct MeshLocalIndexTag { enum { enabled = true }; };

// Config tag specifying the MeshConfig to use.
template< typename ConfigTag >
struct MeshConfigTemplateTag
{
   template< typename Cell, int WorldDimension, typename Real, typename GlobalIndex, typename LocalIndex >
   using MeshConfig = DefaultConfig< Cell, WorldDimension, Real, GlobalIndex, LocalIndex >;
};

// The Mesh is enabled for allowed Device, CellTopology, WorldDimension, Real,
// GlobalIndex, LocalIndex and Id types as specified above.
//
// By specializing this tag you can enable or disable custom combinations of
// the grid template parameters. The default configuration is identical to the
// individual per-type tags.
//
// NOTE: We can't specialize the whole MeshType as it was done for the GridTag,
//       because we don't know the MeshConfig and the compiler can't deduce it
//       at the time of template specializations, so something like this does
//       not work:
//
//          struct MeshTag< ConfigTag,
//                      Mesh< typename MeshConfigTemplateTag< ConfigTag >::
//                         template MeshConfig< CellTopology, WorldDimension, Real, GlobalIndex, LocalIndex > > >
//
template< typename ConfigTag, typename Device, typename CellTopology, int WorldDimension, typename Real, typename GlobalIndex, typename LocalIndex >
struct MeshTag
{
   enum { enabled =
            MeshDeviceTag< ConfigTag, Device >::enabled &&
            MeshCellTopologyTag< ConfigTag, CellTopology >::enabled &&
            MeshWorldDimensionTag< ConfigTag, CellTopology, WorldDimension >::enabled &&
            MeshRealTag< ConfigTag, Real >::enabled &&
            MeshGlobalIndexTag< ConfigTag, GlobalIndex >::enabled &&
            MeshLocalIndexTag< ConfigTag, LocalIndex >::enabled
   };
};

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL
