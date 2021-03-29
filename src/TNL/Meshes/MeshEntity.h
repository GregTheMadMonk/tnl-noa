/***************************************************************************
                          MeshEntity.h  -  description
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

#include <TNL/Meshes/Mesh.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename Device,
          typename EntityTopology_ >
class MeshEntity
{
   static_assert( std::is_same< EntityTopology_, typename Mesh< MeshConfig, Device >::template EntityTraits< EntityTopology_::dimension >::EntityTopology >::value,
                  "Specified entity topology is not compatible with the MeshConfig." );

   public:
      using MeshType        = Mesh< MeshConfig, Device >;
      using DeviceType      = Device;
      using EntityTopology  = EntityTopology_;
      using GlobalIndexType = typename MeshType::GlobalIndexType;
      using LocalIndexType  = typename MeshType::LocalIndexType;
      using PointType       = typename MeshType::PointType;
      using TagType         = typename MeshType::MeshTraitsType::EntityTagType;

      template< int Subdimension >
      using SubentityTraits = typename MeshType::MeshTraitsType::template SubentityTraits< EntityTopology, Subdimension >;

      template< int Superdimension >
      using SuperentityTraits = typename MeshType::MeshTraitsType::template SuperentityTraits< EntityTopology, Superdimension >;

      // constructors
      MeshEntity() = delete;

      __cuda_callable__
      MeshEntity( const MeshType& mesh, const GlobalIndexType index );

      __cuda_callable__
      MeshEntity( const MeshEntity& entity ) = default;

      __cuda_callable__
      MeshEntity& operator=( const MeshEntity& entity ) = default;

      __cuda_callable__
      bool operator==( const MeshEntity& entity ) const;

      __cuda_callable__
      bool operator!=( const MeshEntity& entity ) const;

      static constexpr int getEntityDimension();

      __cuda_callable__
      const MeshType& getMesh() const;

      __cuda_callable__
      GlobalIndexType getIndex() const;

      /****
       * Points
       */
      __cuda_callable__
      PointType getPoint() const;

      /****
       * Subentities
       */
      template< int Subdimension >
      __cuda_callable__
      LocalIndexType getSubentitiesCount() const;

      template< int Subdimension >
      __cuda_callable__
      GlobalIndexType getSubentityIndex( const LocalIndexType localIndex ) const;

      /****
       * Superentities
       */
      template< int Superdimension >
      __cuda_callable__
      LocalIndexType getSuperentitiesCount() const;

      template< int Superdimension >
      __cuda_callable__
      GlobalIndexType getSuperentityIndex( const LocalIndexType localIndex ) const;

      /****
       * Tags
       */
      __cuda_callable__
      TagType getTag() const;

   protected:
      const MeshType* meshPointer = nullptr;
      GlobalIndexType index = 0;
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
std::ostream& operator<<( std::ostream& str, const MeshEntity< MeshConfig, Device, EntityTopology >& entity );

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/MeshEntity.hpp>
