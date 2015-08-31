/***************************************************************************
                          tnlMeshSubentityTraits.h  -  description
                             -------------------
    begin                : Feb 12, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLMESHSUBENTITYTRAITS_H_
#define TNLMESHSUBENTITYTRAITS_H_

#include <core/arrays/tnlStaticArray.h>
#include <core/arrays/tnlSharedArray.h>
#include <mesh/tnlMeshEntity.h>
#include <mesh/config/tnlMeshConfigBase.h>
#include <mesh/topologies/tnlMeshEntityTopology.h>


template< typename MeshConfig, typename EntityTopology > class tnlMeshEntityOrientation;

template< typename MeshConfig,
          typename EntityTag,
          int Dimensions >
class tnlMeshSubentityTraits
{
   public:   
      static const bool storageEnabled = MeshConfig::subentityStorage( EntityTag(), Dimensions );
      static const bool orientationEnabled = MeshConfig::subentityOrientationStorage( EntityTag(), Dimensions );
      
      //typedef tnlStorageTraits< storageEnabled >                    SubentityStorageTag;

      typedef typename MeshConfig::GlobalIndexType                  GlobalIndexType;
      typedef typename MeshConfig::LocalIndexType                   LocalIndexType;
      typedef tnlMeshSubtopology< EntityTag, Dimensions > Tag;


      typedef tnlMeshEntity< MeshConfig, EntityTag >                 EntityType;
      typedef typename Tag::Topology                                     SubentityTag;
      typedef tnlMeshEntity< MeshConfig, SubentityTag >              SubentityType;
      typedef tnlMeshEntitySeed< MeshConfig, SubentityTag >          Seed;
      typedef tnlMeshEntityOrientation< MeshConfig, SubentityTag >   Orientation;


      enum { count = Tag::count };

      typedef tnlStaticArray< count, GlobalIndexType >              ContainerType;
      typedef tnlSharedArray< GlobalIndexType,
                              tnlHost,
                              LocalIndexType >                      SharedContainerType;
      typedef tnlStaticArray< count, GlobalIndexType >              IdArrayType;
      typedef tnlStaticArray< count, SubentityType >                SubentityContainerType;
      typedef tnlStaticArray< count, Seed >                         SeedArrayType;
      typedef tnlStaticArray< count, Orientation >                  OrientationArrayType;
      typedef tnlStaticArray< count, LocalIndexType >               IdPermutationArrayType;

      template< LocalIndexType subentityIndex,
                LocalIndexType subentityVertexIndex >
      struct Vertex
      {
         enum { index = tnlSubentityVertex< EntityTag,
                                            SubentityTag,
                                            subentityIndex,
                                            subentityVertexIndex>::index };
      };

      static_assert( EntityTag::dimensions > Dimensions, "You try to create subentities traits where subentity dimensions are not smaller than the entity dimensions." );
};



#endif /* TNLMESHSUBENTITYTRAITS_H_ */
