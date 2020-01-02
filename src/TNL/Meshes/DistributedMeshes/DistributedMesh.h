/***************************************************************************
                          DistributedMesh.h  -  description
                             -------------------
    begin                : March 17, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/StaticVector.h>

namespace TNL {
namespace Meshes { 
namespace DistributedMeshes {

template< typename MeshType >    
class DistributedMesh
{
public:
   // FIXME: this is not going to work
   using SubdomainOverlapsType = Containers::StaticVector< MeshType::getMeshDimension(), typename MeshType::GlobalIndexType >;

   bool IsDistributed(void)
   {
      return false;
   };

   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix )
   {
      return false;
   }
};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/DistributedMeshes/DistributedGrid.h>
