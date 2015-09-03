/***************************************************************************
                          tnlMeshIntegrityChecker.h  -  description
                             -------------------
    begin                : Mar 20, 2014
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

#ifndef TNLMESHINTEGRITYCHECKER_H_
#define TNLMESHINTEGRITYCHECKER_H_

#include <mesh/tnlMesh.h>
#include <mesh/tnlMeshIntegrityCheckerLayer.h>

template< typename MeshType >
class tnlMeshIntegrityChecker
: public tnlMeshIntegrityCheckerLayer< MeshType,
                                       tnlDimensionsTraits< MeshType::Config::CellTag::dimensions > >
{
      typedef tnlDimensionsTraits< MeshType::Config::CellTag::dimensions > DimensionsTraits;
      typedef tnlMeshIntegrityCheckerLayer< MeshType, DimensionsTraits > BaseType;

   public:
      static bool checkMesh( const MeshType& mesh )
      {
         if( ! BaseType::checkEntities( mesh ) )
            return false;
         return true;
      }
};


#endif /* TNLMESHINTEGRITYCHECKER_H_ */
