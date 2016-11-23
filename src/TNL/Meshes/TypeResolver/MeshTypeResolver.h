/***************************************************************************
                          MeshTypeResolver.h  -  description
                             -------------------
    begin                : Nov 22, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>

#include <TNL/Meshes/BuildConfigTags.h>

namespace TNL {
namespace Meshes {

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
class MeshTypeResolver
{
public:

   static bool run( Reader& reader,
                    ProblemSetterArgs&&... problemSetterArgs );

protected:

   static bool resolveCellTopology( Reader& reader,
                                    ProblemSetterArgs&&... problemSetterArgs );

   // NOTE: We could disable the meshes only by the MeshTag, but doing the
   //       resolution for all subtypes is more flexible and also pretty
   //       good optimization of compilation times.

   // Overload for disabled cell topologies
   template< typename CellTopology,
             typename = typename std::enable_if< ! BuildConfigTags::MeshCellTopologyTag< ConfigTag, CellTopology >::enabled >::type,
             typename = void >
   static bool resolveWorldDimension( Reader& reader,
                                      ProblemSetterArgs&&... problemSetterArgs );

   // Overload for enabled cell topologies
   template< typename CellTopology,
             typename = typename std::enable_if< BuildConfigTags::MeshCellTopologyTag< ConfigTag, CellTopology >::enabled >::type >
   static bool resolveWorldDimension( Reader& reader,
                                      ProblemSetterArgs&&... problemSetterArgs );

   // Overload for disabled world dimensions
   template< typename CellTopology,
             int WorldDimension,
             typename = typename std::enable_if< ! BuildConfigTags::MeshWorldDimensionTag< ConfigTag, CellTopology, WorldDimension >::enabled >::type,
             typename = void >
   static bool resolveReal( Reader& reader,
                            ProblemSetterArgs&&... problemSetterArgs );

   // Overload for enabled world dimensions
   template< typename CellTopology,
             int WorldDimension,
             typename = typename std::enable_if< BuildConfigTags::MeshWorldDimensionTag< ConfigTag, CellTopology, WorldDimension >::enabled >::type >
   static bool resolveReal( Reader& reader,
                            ProblemSetterArgs&&... problemSetterArgs );

   // Overload for disabled real types
   template< typename CellTopology,
             int WorldDimension,
             typename Real,
             typename = typename std::enable_if< ! BuildConfigTags::MeshRealTag< ConfigTag, Real >::enabled >::type,
             typename = void >
   static bool resolveGlobalIndex( Reader& reader,
                                   ProblemSetterArgs&&... problemSetterArgs );

   // Overload for enabled real types
   template< typename CellTopology,
             int WorldDimension,
             typename Real,
             typename = typename std::enable_if< BuildConfigTags::MeshRealTag< ConfigTag, Real >::enabled >::type >
   static bool resolveGlobalIndex( Reader& reader,
                                   ProblemSetterArgs&&... problemSetterArgs );

   // Overload for disabled global index types
   template< typename CellTopology,
             int WorldDimension,
             typename Real,
             typename GlobalIndex,
             typename = typename std::enable_if< ! BuildConfigTags::MeshGlobalIndexTag< ConfigTag, GlobalIndex >::enabled >::type,
             typename = void >
   static bool resolveLocalIndex( Reader& reader,
                                  ProblemSetterArgs&&... problemSetterArgs );

   // Overload for enabled global index types
   template< typename CellTopology,
             int WorldDimension,
             typename Real,
             typename GlobalIndex,
             typename = typename std::enable_if< BuildConfigTags::MeshGlobalIndexTag< ConfigTag, GlobalIndex >::enabled >::type >
   static bool resolveLocalIndex( Reader& reader,
                                  ProblemSetterArgs&&... problemSetterArgs );

   // Overload for disabled local index types
   template< typename CellTopology,
             int WorldDimension,
             typename Real,
             typename GlobalIndex,
             typename LocalIndex,
             typename = typename std::enable_if< ! BuildConfigTags::MeshLocalIndexTag< ConfigTag, LocalIndex >::enabled >::type,
             typename = void >
   static bool resolveId( Reader& reader,
                          ProblemSetterArgs&&... problemSetterArgs );

   // Overload for enabled local index types
   template< typename CellTopology,
             int WorldDimension,
             typename Real,
             typename GlobalIndex,
             typename LocalIndex,
             typename = typename std::enable_if< BuildConfigTags::MeshLocalIndexTag< ConfigTag, LocalIndex >::enabled >::type >
   static bool resolveId( Reader& reader,
                          ProblemSetterArgs&&... problemSetterArgs );

   // Overload for disabled id types
   template< typename CellTopology,
             int WorldDimension,
             typename Real,
             typename GlobalIndex,
             typename LocalIndex,
             typename Id,
             typename = typename std::enable_if< ! BuildConfigTags::MeshIdTag< ConfigTag, GlobalIndex, Id >::enabled >::type,
             typename = void >
   static bool resolveMeshType( Reader& reader,
                                ProblemSetterArgs&&... problemSetterArgs );

   // Overload for enabled id types
   template< typename CellTopology,
             int WorldDimension,
             typename Real,
             typename GlobalIndex,
             typename LocalIndex,
             typename Id,
             typename = typename std::enable_if< BuildConfigTags::MeshIdTag< ConfigTag, GlobalIndex, Id >::enabled >::type >
   static bool resolveMeshType( Reader& reader,
                                ProblemSetterArgs&&... problemSetterArgs );

   // Overload for disabled mesh types
   template< typename MeshType,
             typename = typename std::enable_if< ! BuildConfigTags::MeshDeviceTag< ConfigTag, Device >::enabled ||
                                                 ! BuildConfigTags::MeshTag< ConfigTag,
                                                                             Device,
                                                                             typename MeshType::Config::CellTopology,
                                                                             MeshType::Config::worldDimensions,
                                                                             typename MeshType::Config::RealType,
                                                                             typename MeshType::Config::GlobalIndexType,
                                                                             typename MeshType::Config::LocalIndexType,
                                                                             typename MeshType::Config::IdType
                                                                           >::enabled >::type,
             typename = void >
   static bool resolveTerminate( Reader& reader,
                                 ProblemSetterArgs&&... problemSetterArgs );

   // Overload for enabled mesh types
   template< typename MeshType,
             typename = typename std::enable_if< BuildConfigTags::MeshDeviceTag< ConfigTag, Device >::enabled &&
                                                 BuildConfigTags::MeshTag< ConfigTag,
                                                                             Device,
                                                                             typename MeshType::Config::CellTopology,
                                                                             MeshType::Config::worldDimensions,
                                                                             typename MeshType::Config::RealType,
                                                                             typename MeshType::Config::GlobalIndexType,
                                                                             typename MeshType::Config::LocalIndexType,
                                                                             typename MeshType::Config::IdType
                                                                           >::enabled >::type >
   static bool resolveTerminate( Reader& reader,
                                 ProblemSetterArgs&&... problemSetterArgs );
};

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/TypeResolver/MeshTypeResolver_impl.h>
