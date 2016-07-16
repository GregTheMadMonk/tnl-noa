/***************************************************************************
                          tnlMeshIntegrityChecker.h  -  description
                             -------------------
    begin                : Mar 20, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLMESHINTEGRITYCHECKER_H_
#define TNLMESHINTEGRITYCHECKER_H_

#include <mesh/tnlMesh.h>
#include <mesh/tnlMeshIntegrityCheckerLayer.h>

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


#endif /* TNLMESHINTEGRITYCHECKER_H_ */
