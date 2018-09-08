/***************************************************************************
                          DistributedMesh.h  -  description
                             -------------------
    begin                : March 17, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Meshes { 
namespace DistributedMeshes {

template< typename MeshType >    
class DistributedMesh
{
public:
    bool IsDistributed(void)
    {
        return false;
    };
};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/DistributedMeshes/DistributedGrid.h>
