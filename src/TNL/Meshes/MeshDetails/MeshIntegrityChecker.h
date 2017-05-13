/***************************************************************************
                          MeshIntegrityChecker.h  -  description
                             -------------------
    begin                : Mar 20, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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
#include <TNL/Meshes/MeshDetails/MeshIntegrityCheckerLayer.h>

namespace TNL {
namespace Meshes {

template< typename MeshType >
class MeshIntegrityChecker
: public MeshIntegrityCheckerLayer< MeshType,
                                       MeshDimensionTag< MeshType::Config::CellType::dimensions > >
{
      typedef MeshDimensionTag< MeshType::Config::CellType::dimensions > DimensionTag;
      typedef MeshIntegrityCheckerLayer< MeshType, DimensionTag > BaseType;

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
