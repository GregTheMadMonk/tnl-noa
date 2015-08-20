/***************************************************************************
                          tnlMeshConfigTraits.h  -  description
                             -------------------
    begin                : Aug 14, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
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

#ifndef TNLMESHCONFIGTRAITS_H
#define	TNLMESHCONFIGTRAITS_H

#include <core/tnlHost.h>
#include <mesh/traits/tnlMeshEntitiesTraits.h>
#include <mesh/traits/tnlMeshSubentitiesTraits.h>
#include <mesh/traits/tnlMeshSuperentitiesTraits.h>

template< typename MeshConfig >
class tnlMeshConfigTraits
{
   public:
      static const int meshDimensions = MeshConfig::CellType::dimensions;
      static const int worldDimensions = MeshConfig::worldDimensions;

      typedef typename MeshConfig::RealType                                     RealType;
      typedef typename MeshConfig::GlobalIndexType                              GlobalIndexType;
      typedef typename MeshConfig::LocalIndexType                               LocalIndexType;
      typedef typename tnlMeshEntitiesTag< MeshConfig, 
                                           tnlDimensionsTag< 0 > >::Tag         VertexType;
      typedef typename MeshConfig::CellType                                     CellType;

      typedef tnlDimensionsTag< meshDimensions >                                MeshDimensions;
      typedef typename tnlMeshTraits< MeshConfig >::PointType                   Point;
      //typedef Entity<TConfig, TVertexTopology>                                  TVertex;     
      typedef tnlMeshEntitySeed< MeshConfig, CellType >                         CellSeed;

      // TODO: to asi presunout do implementace, stejne to bude vazane na TNL
      typedef tnlArray< GlobalIndexType, tnlHost, GlobalIndexType >             GlobalIdArrayType;
      typedef tnlSharedArray< GlobalIndexType, tnlHost, LocalIndexType >        IdArrayAccessorType;
      typedef tnlSharedArray< LocalIndexType, tnlHost, LocalIndexType >         IdPermutationArrayAccessorType;
      typedef tnlArray< Point, tnlHost, GlobalIndexType>                        PointArrayType;
      typedef tnlArray< CellSeed, tnlHost, GlobalIndexType >                    CellSeedArrayType;

      template< typename Dimensions > using EntityTraits = 
         tnlMeshEntitiesTraits< MeshConfig, Dimensions >;
      
      template< typename MeshEntity, typename SubDimensions > using SubentityTraits =
         tnlMeshSubentitiesTraits< MeshConfig, MeshEntity, SubDimensions >;
      
      template< typename MeshEntity, typename SuperDimensions > using SuperentityTraits =
         tnlMeshSuperentitiesTraits< MeshConfig, MeshEntity, SuperDimensions >;
};

#endif	/* TNLMESHCONFIGTRAITS_H */

