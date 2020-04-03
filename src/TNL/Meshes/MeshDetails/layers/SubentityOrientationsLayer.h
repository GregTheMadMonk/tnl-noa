/***************************************************************************
                          SubentityOrientationsLayer.h  -  description
                             -------------------
    begin                : Mar 24, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/DimensionTag.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/MeshDetails/traits/WeakStorageTraits.h>
#include <TNL/Meshes/MeshDetails/MeshEntityOrientation.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SubdimensionTag,
          bool SubentityOrientationStorage =
               WeakSubentityStorageTrait< MeshConfig, Device, EntityTopology, SubdimensionTag >::storageEnabled &&
               MeshConfig::subentityOrientationStorage( EntityTopology(), SubdimensionTag::value ) >
class SubentityOrientationsLayer;


template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
class SubentityOrientationsLayerFamily
   : public SubentityOrientationsLayer< MeshConfig,
                                        Device,
                                        EntityTopology,
                                        Meshes::DimensionTag< 0 > >
{
   using BaseType = SubentityOrientationsLayer< MeshConfig,
                                                 Device,
                                                 EntityTopology,
                                                 Meshes::DimensionTag< 0 > >;
public:
   using BaseType::subentityOrientationsArray;
   using BaseType::getSubentityOrientation;
};


template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SubdimensionTag >
class SubentityOrientationsLayer< MeshConfig,
                                  Device,
                                  EntityTopology,
                                  SubdimensionTag,
                                  true >
   : public SubentityOrientationsLayer< MeshConfig,
                                        Device,
                                        EntityTopology,
                                        typename SubdimensionTag::Increment >
{
   using BaseType = SubentityOrientationsLayer< MeshConfig,
                                                 Device,
                                                 EntityTopology,
                                                 typename SubdimensionTag::Increment >;

   using MeshTraitsType         = MeshTraits< MeshConfig, Device >;
   using SubentityTraitsType    = typename MeshTraitsType::template SubentityTraits< EntityTopology, SubdimensionTag::value >;
   using DimensionTag           = Meshes::DimensionTag< EntityTopology::dimension >;

protected:
   using GlobalIndexType        = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType         = typename MeshTraitsType::LocalIndexType;
   using OrientationArrayType   = typename SubentityTraitsType::OrientationArrayType;
   using IdPermutationArrayType = typename SubentityTraitsType::IdPermutationArrayType;
   using OrientationsStorageArrayType = typename SubentityTraitsType::OrientationsStorageArrayType;

   void setEntitiesCount( const GlobalIndexType entitiesCount )
   {
      orientations.setSize( entitiesCount );
      BaseType::setEntitiesCount( entitiesCount );
   }

   using BaseType::getSubentityOrientation;
   __cuda_callable__
   const IdPermutationArrayType& getSubentityOrientation( DimensionTag, SubdimensionTag, GlobalIndexType entityIndex, LocalIndexType localIndex ) const
   {
      return orientations[ entityIndex ][ localIndex ].getSubvertexPermutation();
   }

   using BaseType::subentityOrientationsArray;
   __cuda_callable__
   OrientationArrayType& subentityOrientationsArray( DimensionTag, SubdimensionTag, GlobalIndexType entityIndex )
   {
      return orientations[ entityIndex ];
   }

private:
   OrientationsStorageArrayType orientations;
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SubdimensionTag >
class SubentityOrientationsLayer< MeshConfig,
                                  Device,
                                  EntityTopology,
                                  SubdimensionTag,
                                  false >
   : public SubentityOrientationsLayer< MeshConfig,
                                        Device,
                                        EntityTopology,
                                        typename SubdimensionTag::Increment >
{
   using BaseType = SubentityOrientationsLayer< MeshConfig,
                                                 Device,
                                                 EntityTopology,
                                                 typename SubdimensionTag::Increment >;

   using MeshTraitsType         = MeshTraits< MeshConfig, Device >;
   using DimensionTag           = Meshes::DimensionTag< EntityTopology::dimension >;

protected:
   using GlobalIndexType        = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType         = typename MeshTraitsType::LocalIndexType;

   void setEntitiesCount( const GlobalIndexType entitiesCount )
   {
      BaseType::setEntitiesCount( entitiesCount );
   }

   using BaseType::getSubentityOrientation;
   using BaseType::subentityOrientationsArray;

   __cuda_callable__
   void getSubentityOrientation( DimensionTag, SubdimensionTag, GlobalIndexType entityIndex, LocalIndexType localIndex ) const {}

   __cuda_callable__
   void subentityOrientationsArray( DimensionTag, SubdimensionTag, GlobalIndexType entityIndex ) {}
};

// termination of recursive inheritance (everything is reduced to EntityStorage == false thanks to the WeakSubentityStorageTrait)
template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
class SubentityOrientationsLayer< MeshConfig,
                            Device,
                            EntityTopology,
                            DimensionTag< EntityTopology::dimension >,
                            false >
{
protected:
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;
   using LocalIndexType  = typename MeshConfig::LocalIndexType;
   using DimensionTag    = Meshes::DimensionTag< EntityTopology::dimension >;
   using SubdimensionTag = Meshes::DimensionTag< EntityTopology::dimension >;

   /***
    * Necessary because of 'using BaseType::...;' in the derived classes
    */
   void setEntitiesCount( GlobalIndexType ) {}
   __cuda_callable__
   void getSubentityOrientation( DimensionTag, SubdimensionTag, GlobalIndexType, LocalIndexType ) const {}
   __cuda_callable__
   void subentityOrientationsArray( DimensionTag, SubdimensionTag, GlobalIndexType ) {}
};

} // namespace Meshes
} // namespace TNL
