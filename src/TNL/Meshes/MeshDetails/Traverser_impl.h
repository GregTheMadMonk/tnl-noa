/***************************************************************************
                          Traverser_impl.h  -  description
                             -------------------
    begin                : Dec 25, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Traverser.h>

namespace TNL {
namespace Meshes {   

template< typename Mesh,
          typename MeshEntity,
          int EntitiesDimension >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Mesh, MeshEntity, EntitiesDimension >::
processBoundaryEntities( const MeshPointer& meshPointer,
                         SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   auto entitiesCount = meshPointer->template getBoundaryEntitiesCount< EntitiesDimension >();
#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
   for( decltype(entitiesCount) i = 0; i < entitiesCount; i++ ) {
      const auto entityIndex = meshPointer->template getBoundaryEntityIndex< EntitiesDimension >( i );
      auto& entity = meshPointer->template getEntity< EntitiesDimension >( entityIndex );
      // TODO: if the Mesh::IdType is void, then we should also pass the entityIndex
      EntitiesProcessor::processEntity( *meshPointer, *userDataPointer, entity );
   }
}

template< typename Mesh,
          typename MeshEntity,
          int EntitiesDimension >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Mesh, MeshEntity, EntitiesDimension >::
processInteriorEntities( const MeshPointer& meshPointer,
                         SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   auto entitiesCount = meshPointer->template getInteriorEntitiesCount< EntitiesDimension >();
#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
   for( decltype(entitiesCount) i = 0; i < entitiesCount; i++ ) {
      const auto entityIndex = meshPointer->template getInteriorEntityIndex< EntitiesDimension >( i );
      auto& entity = meshPointer->template getEntity< EntitiesDimension >( entityIndex );
      // TODO: if the Mesh::IdType is void, then we should also pass the entityIndex
      EntitiesProcessor::processEntity( *meshPointer, *userDataPointer, entity );
   }
}

template< typename Mesh,
          typename MeshEntity,
          int EntitiesDimension >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Mesh, MeshEntity, EntitiesDimension >::
processAllEntities( const MeshPointer& meshPointer,
                    SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   auto entitiesCount = meshPointer->template getEntitiesCount< EntitiesDimension >();
#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
   for( decltype(entitiesCount) entityIndex = 0; entityIndex < entitiesCount; entityIndex++ ) {
      auto& entity = meshPointer->template getEntity< EntitiesDimension >( entityIndex );
      // TODO: if the Mesh::IdType is void, then we should also pass the entityIndex
      EntitiesProcessor::processEntity( *meshPointer, *userDataPointer, entity );
   }
}

} // namespace Meshes
} // namespace TNL
