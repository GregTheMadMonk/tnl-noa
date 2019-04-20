/***************************************************************************
                          DistributedMeshSynchronizer.h  -  description
                             -------------------
    begin                : January 8, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Meshes {
namespace DistributedMeshes {

template <typename MeshFunctionType>
class DistributedMeshSynchronizer
{
   public:

      // FIXME: clang does not like this (incomplete type error)
//      typedef typename MeshFunctionType::DistributedMeshType DistributedMeshType;

      template< typename DistributedMeshType >
      void setDistributedGrid( DistributedMeshType *distributedGrid )
      {
         throw Exceptions::NotImplementedError("Distributed version of this mesh type is not implemented.");
      }
};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/DistributedMeshes/DistributedGridSynchronizer.h>
