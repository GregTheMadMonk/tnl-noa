/***************************************************************************
                          tnlMeshIntegrityChecker.h  -  description
                             -------------------
    begin                : Mar 20, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <mesh/tnlMesh.h>
#include <mesh/tnlMeshIntegrityCheckerLayer.h>

namespace TNL {

template< typename MeshType >
class tnlMeshIntegrityChecker
: public tnlMeshIntegrityCheckerLayer< MeshType,
                                       tnlDimensionsTag< MeshType::Config::CellType::dimensions > >
{
      typedef tnlDimensionsTag< MeshType::Config::CellType::dimensions > DimensionsTag;
      typedef tnlMeshIntegrityCheckerLayer< MeshType, DimensionsTag > BaseType;

   public:
      static bool checkMesh( const MeshType& mesh )
      {
         if( ! BaseType::checkEntities( mesh ) )
            return false;
         return true;
      }
};

} // namespace TNL
