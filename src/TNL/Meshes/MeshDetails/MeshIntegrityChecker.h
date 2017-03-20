/***************************************************************************
                          MeshIntegrityChecker.h  -  description
                             -------------------
    begin                : Mar 20, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshDetails/MeshIntegrityCheckerLayer.h>

namespace TNL {
namespace Meshes {

template< typename MeshType >
class MeshIntegrityChecker
: public MeshIntegrityCheckerLayer< MeshType,
                                       MeshDimensionsTag< MeshType::Config::CellType::dimensions > >
{
      typedef MeshDimensionsTag< MeshType::Config::CellType::dimensions > DimensionsTag;
      typedef MeshIntegrityCheckerLayer< MeshType, DimensionsTag > BaseType;

   public:
      static bool checkMesh( const MeshType& mesh )
      {
         if( ! BaseType::checkEntities( mesh ) )
            return false;
         return true;
      }
};

} // namespace Meshes
} // namespace TNL
