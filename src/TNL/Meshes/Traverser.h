/***************************************************************************
                          Traverser.h  -  description
                             -------------------
    begin                : Jul 28, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Meshes/Mesh.h>

namespace TNL {
namespace Meshes {

template< typename Mesh,
          typename MeshEntity,
          int EntitiesDimension = MeshEntity::getEntityDimension() >
class Traverser
{
   public:
      using MeshType = Mesh;
      using MeshPointer = Pointers::SharedPointer<  MeshType >;
      using DeviceType = typename MeshType::DeviceType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const MeshPointer& meshPointer,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const MeshPointer& meshPointer,
                                       UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const MeshPointer& meshPointer,
                                 UserData& userData ) const;
};

template< typename MeshConfig,
          typename MeshEntity,
          int EntitiesDimension >
class Traverser< Mesh< MeshConfig, Devices::Cuda >, MeshEntity, EntitiesDimension >
{
   public:
      using MeshType = Mesh< MeshConfig, Devices::Cuda >;
      using MeshPointer = Pointers::SharedPointer<  MeshType >;
      using DeviceType = typename MeshType::DeviceType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const MeshPointer& meshPointer,
                                       UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const MeshPointer& meshPointer,
                                       UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const MeshPointer& meshPointer,
                                 UserData& userData ) const;
};

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/MeshDetails/Traverser_impl.h>
#include <TNL/Meshes/GridDetails/Traverser_Grid1D.h>
#include <TNL/Meshes/GridDetails/Traverser_Grid2D.h>
#include <TNL/Meshes/GridDetails/Traverser_Grid3D.h>
